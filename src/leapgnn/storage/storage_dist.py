# Vendored from LeapGNN-AE/storage/storage_dist.py
# Changes vs. original:
#   1. DGL imports guarded with try/except (DGL not required in DistEA).
#   2. `import data` removed (LeapGNN-AE internal module, not present in DistEA).
#   3. `from rpc_client import ...` → `from ..rpc_client import ...` (package-relative).

import os

import numpy as np
import torch

try:
    import dgl.utils
    from dgl import DGLGraph
    from dgl.frame import Frame, FrameRef

    _HAS_DGL = True
except (ImportError, FileNotFoundError):
    _HAS_DGL = False

import logging
import mmap
import time

import grpc
import torch.nn.functional as F

from ..rpc_client import distcache_pb2, distcache_pb2_grpc


class DistCacheClient:
    def __init__(self, grpc_port, gpu, log):
        self.grpc_port = grpc_port
        # connect to the cache server
        self.channel = grpc.insecure_channel(
            self.grpc_port,
            options=[
                ("grpc.enable_retries", 1),
                ("grpc.keepalive_timeout_ms", 100000),
                (
                    "grpc.max_receive_message_length",
                    2000 * 1024 * 1024,
                ),  # max grpc size 2GB
            ],
        )
        # client can use this stub to request to golang cache server
        self.stub = distcache_pb2_grpc.OperatorStub(self.channel)

        # open the stream-features connection to the cache server
        self.channel_features = grpc.insecure_channel(
            self.grpc_port,
            options=[
                ("grpc.enable_retries", 1),
                ("grpc.keepalive_timeout_ms", 100000),
                (
                    "grpc.max_receive_message_length",
                    2000 * 1024 * 1024,
                ),  # max grpc size 2GB
            ],
        )
        # client can use this stub to request stream features to golang cache server
        self.stub_features = distcache_pb2_grpc.OperatorFeaturesStub(
            self.channel_features
        )

        self.gpuid = gpu
        self.feat_dim = self.get_feat_dim()
        self.dims = {"features": self.feat_dim}
        self.log = log
        self.server_log = self.get_statistic()
        assert self.log == self.server_log, "client and server log flags differ"

        self.try_num = 0
        self.miss_num = 0

        self.cnt = 0
        # self.feats_chunk_size = 4*1024*1024 # 4MB
        self.feats_chunk_size = self.getMaxNumDivisibleByXYAnd1024(self.feat_dim, 4)
        print(f"-> feats chunk size: {self.feats_chunk_size}")

        # rm /dev/shm/repgnn_shm*
        prefix = "repgnn_shm"
        if any(filename.startswith(prefix) for filename in os.listdir("/dev/shm")):
            os.system("rm " + os.path.join(" /dev/shm", prefix) + "*")

    def ConstructNid2Pid(
        self, datasetpath, partition_num, partition_type, total_graph_nodes
    ):
        # construct Nid2pid
        self.Nid2pid = np.zeros(total_graph_nodes)
        partition_path = os.path.join(
            datasetpath, f"dist_True/{partition_num}_{partition_type}/"
        )
        npy_files = [
            file for file in os.listdir(partition_path) if file.endswith(".npy")
        ]
        world_size = 0
        for file_name in npy_files:
            world_size += 1
            pid = int(file_name.split(".")[0])
            file_path = os.path.join(partition_path, file_name)
            tmp_array = np.load(file_path)
            self.Nid2pid[tmp_array] = pid
        self.world_size = world_size

    def fetch_data(self, nodeflow):
        with torch.autograd.profiler.record_function("get nf_nids"):
            # load the sub-graph lnids onto the GPU; node_mapping maps nf-level -> part-graph lnid
            nf_nids = nodeflow._node_mapping.tousertensor()
            offsets = nodeflow._layer_offsets
            logging.debug(f"fetch_data batch onid, layer_offset: {nf_nids}, {offsets}")
        logging.debug(f"nf.nlayers: {nodeflow.num_layers}")
        for i in range(nodeflow.num_layers):
            tnid = nf_nids[offsets[i] : offsets[i + 1]]
            # create frame
            with torch.autograd.profiler.record_function("create frames"):
                frame = {
                    name: torch.empty(tnid.size(0), self.dims[name])
                    for name in self.dims
                }
                tnid = tnid.tolist()
            with torch.autograd.profiler.record_function(
                "fetch feat from cache server"
            ):
                row_st_idx, row_ed_idx = 0, 0
                for sub_features in self.get_stream_feats_from_server_v2(tnid):
                    for name in self.dims:
                        sub_tensor = torch.frombuffer(
                            sub_features, dtype=torch.float32
                        ).reshape(-1, self.feat_dim)
                        row_ed_idx += sub_tensor.shape[0]
                        frame[name][row_st_idx:row_ed_idx, :] = sub_tensor
                        row_st_idx = row_ed_idx
            with torch.autograd.profiler.record_function("move feats from CPU to GPU"):
                for name in self.dims:
                    frame[name].data = frame[name].data.cuda(self.gpuid)
            with torch.autograd.profiler.record_function("asign frame to nodeflow"):
                logging.debug(
                    f'Final nodeflow._node_frames:{i}, frame["features"].size(): {frame["features"].size()}\n'
                )
                if _HAS_DGL:
                    nodeflow._node_frames[i] = FrameRef(Frame(frame))

    def fetch_multiple_nfs(self, nfs):
        world_size = len(nfs)
        with torch.autograd.profiler.record_function("get nf_nids"):
            nf_nids = []
            offsets = []
            for j in range(world_size):
                nf_nids.append(nfs[j]._node_mapping.tousertensor().cuda(self.gpuid))
                offsets.append(nfs[j]._layer_offsets)

        for i in range(nfs[0].num_layers):
            tnid = []
            tnids_flat = []
            for j in range(world_size):
                tnid.append(nf_nids[j][offsets[j][i] : offsets[j][i + 1]])
                tnid[j] = tnid[j].tolist()
                tnids_flat.extend(tnid[j])

            n_rows = []
            with torch.autograd.profiler.record_function("fetch feat overhead"):
                with torch.cuda.device(self.gpuid):
                    frames = []
                    for j in range(world_size):
                        frames.append(
                            {
                                name: torch.empty(len(tnid[j]), self.dims[name])
                                for name in self.dims
                            }
                        )
                        n_rows.extend([len(tnid[j]) for name in self.dims])
            frame_id, row_st_idx, row_ed_idx = 0, 0, 0
            with torch.autograd.profiler.record_function(
                "fetch feat from cache server"
            ):
                st = time.time()
                for sub_features in self.get_stream_feats_from_server(tnids_flat):
                    self.tmp_value += time.time() - st
                    st = time.time()
                    for name in self.dims:
                        sub_tensor = torch.frombuffer(
                            sub_features, dtype=torch.float32
                        ).reshape(-1, self.feat_dim)
                        n_sub_tensor_row = sub_tensor.shape[0]
                    self.tmp_value2 += time.time() - st

                    with torch.autograd.profiler.record_function(
                        "fetch feat from cache server - python while"
                    ):
                        while n_sub_tensor_row:
                            cur_frame_remains_rows = n_rows[frame_id] - row_ed_idx
                            fill_rows = min(cur_frame_remains_rows, n_sub_tensor_row)
                            row_ed_idx += fill_rows
                            frames[frame_id][name][row_st_idx:row_ed_idx, :] = (
                                sub_tensor[:fill_rows, :]
                            )
                            sub_tensor = sub_tensor[fill_rows:, :]
                            n_sub_tensor_row -= fill_rows
                            if n_sub_tensor_row > 0 or row_ed_idx == n_rows[frame_id]:
                                frame_id += 1
                                row_st_idx = row_ed_idx = 0
                            else:
                                row_st_idx = row_ed_idx
                    st = time.time()
            with torch.autograd.profiler.record_function("move feats from CPU to GPU"):
                for j in range(world_size):
                    for name in self.dims:
                        frames[j][name].data = frames[j][name].data.cuda(self.gpuid)
            with torch.autograd.profiler.record_function("asign frame to nodeflow"):
                for j in range(world_size):
                    logging.debug(
                        f'Final nfs._node_frames:{i}, frames[j]["features"].size(): {frames[j]["features"].size()}\n'
                    )
                    if _HAS_DGL:
                        nfs[j]._node_frames[i] = FrameRef(Frame(frames[j]))

    def fetch_multiple_nfs_elimredun(self, nfs):
        world_size = len(nfs)
        with torch.autograd.profiler.record_function("get nf_nids"):
            nf_nids = []
            offsets = []
            for j in range(world_size):
                nf_nids.append(nfs[j]._node_mapping.tousertensor().cuda(self.gpuid))
                offsets.append(nfs[j]._layer_offsets)

        for i in range(nfs[0].num_layers):
            tnid = []
            offs = [0]
            for j in range(world_size):
                tnid.append(nf_nids[j][offsets[j][i] : offsets[j][i + 1]])
                tnid[j] = tnid[j].tolist()
                offs.append(offs[j] + len(tnid[j]))

            with torch.autograd.profiler.record_function("deduplicate tnids"):
                mapping_idx = []
                unique_tnids_flat, index = self.merge_lists(tnid)
                unique_tnids_flat = unique_tnids_flat.tolist()
                for idx in range(len(offs) - 1):
                    mapping_idx.append(index[offs[idx] : offs[idx + 1]])

            n_rows = []
            with torch.autograd.profiler.record_function("create frames"):
                with torch.cuda.device(self.gpuid):
                    frames = []
                    for j in range(world_size):
                        frames.append(
                            {
                                name: torch.empty(len(tnid[j]), self.dims[name])
                                for name in self.dims
                            }
                        )
                        n_rows.extend([len(tnid[j]) for name in self.dims])
            tmp_tensor_lst = []
            with torch.autograd.profiler.record_function(
                "fetch feat from cache server"
            ):
                for sub_features in self.get_stream_feats_from_server_v2(
                    unique_tnids_flat
                ):
                    sub_tensor = torch.frombuffer(
                        sub_features, dtype=torch.float32
                    ).reshape(-1, self.feat_dim)
                    tmp_tensor_lst.append(sub_tensor)
            with torch.autograd.profiler.record_function("concat all fetched feats"):
                fetched_unique_tensor = torch.cat(tmp_tensor_lst, dim=0)
            with torch.autograd.profiler.record_function("fill frames"):
                for j in range(world_size):
                    for name in self.dims:
                        frames[j][name][:, :] = fetched_unique_tensor[mapping_idx[j]]
            with torch.autograd.profiler.record_function("move feats from CPU to GPU"):
                for j in range(world_size):
                    for name in self.dims:
                        frames[j][name].data = frames[j][name].data.cuda(self.gpuid)
            with torch.autograd.profiler.record_function("asign frame to nodeflow"):
                for j in range(world_size):
                    logging.debug(
                        f'Final nfs._node_frames:{i}, frames[j]["features"].size(): {frames[j]["features"].size()}\n'
                    )
                    if _HAS_DGL:
                        nfs[j]._node_frames[i] = FrameRef(Frame(frames[j]))

    def fetch_multiple_nfs_v2(self, nfs):
        world_size = len(nfs)
        with torch.autograd.profiler.record_function("get nf_nids"):
            nf_nids = []
            offsets = []
            for j in range(world_size):
                nf_nids.append(nfs[j]._node_mapping.tousertensor().cuda(self.gpuid))
                offsets.append(nfs[j]._layer_offsets)

        for i in range(nfs[0].num_layers):
            tnid = []
            offs = [0]
            for j in range(world_size):
                tnid.append(nf_nids[j][offsets[j][i] : offsets[j][i + 1]])
                tnid[j] = tnid[j].tolist()
                offs.append(offs[j] + len(tnid[j]))

            with torch.autograd.profiler.record_function("deduplicate tnids"):
                mapping_idx = []
                unique_tnids_flat, index = self.merge_lists_wodedupli(tnid)
                unique_tnids_flat = unique_tnids_flat.tolist()
                for idx in range(len(offs) - 1):
                    mapping_idx.append(index[offs[idx] : offs[idx + 1]])

            n_rows = []
            with torch.autograd.profiler.record_function("create frames"):
                with torch.cuda.device(self.gpuid):
                    frames = []
                    for j in range(world_size):
                        frames.append(
                            {
                                name: torch.empty(len(tnid[j]), self.dims[name])
                                for name in self.dims
                            }
                        )
                        n_rows.extend([len(tnid[j]) for name in self.dims])
            tmp_tensor_lst = []
            with torch.autograd.profiler.record_function(
                "fetch feat from cache server"
            ):
                for sub_features in self.get_stream_feats_from_server(
                    unique_tnids_flat
                ):
                    sub_tensor = torch.frombuffer(
                        sub_features, dtype=torch.float32
                    ).reshape(-1, self.feat_dim)
                    tmp_tensor_lst.append(sub_tensor)
            with torch.autograd.profiler.record_function("concat all fetched feats"):
                fetched_unique_tensor = torch.cat(tmp_tensor_lst, dim=0)
            with torch.autograd.profiler.record_function("fill frames"):
                for j in range(world_size):
                    for name in self.dims:
                        frames[j][name][:, :] = fetched_unique_tensor[mapping_idx[j]]
            with torch.autograd.profiler.record_function("move feats from CPU to GPU"):
                for j in range(world_size):
                    for name in self.dims:
                        frames[j][name].data = frames[j][name].data.cuda(self.gpuid)
            with torch.autograd.profiler.record_function("asign frame to nodeflow"):
                for j in range(world_size):
                    logging.debug(
                        f'Final nfs._node_frames:{i}, frames[j]["features"].size(): {frames[j]["features"].size()}\n'
                    )
                    if _HAS_DGL:
                        nfs[j]._node_frames[i] = FrameRef(Frame(frames[j]))

    def construct_local_cached_dict(
        self, replace_prob, total_vertices, datasetpath, dist_num, rank
    ):
        self.replace_prob = replace_prob
        self.total_vertices = total_vertices
        self.datasetpath = datasetpath
        self.dist_num = dist_num
        self.rank = rank

        local_cached_path = os.path.join(
            datasetpath, f"dist_True/{dist_num}_metis/{rank}.npy"
        )
        local_cached_vertices = np.load(local_cached_path)
        self.whether_local = np.zeros(total_vertices, dtype=bool)
        self.whether_local[local_cached_vertices] = True

    def get_feat_dim(self):
        response = self.stub.DCSubmit(
            distcache_pb2.DCRequest(type=distcache_pb2.get_feature_dim), timeout=1000
        )
        return response.featdim

    def Reset(self):
        response = self.stub.DCSubmit(
            distcache_pb2.DCRequest(type=distcache_pb2.reset), timeout=1000
        )

    def get_statistic(self):
        response = self.stub.DCSubmit(
            distcache_pb2.DCRequest(type=distcache_pb2.get_statistic), timeout=1000
        )
        return response.statistic

    def get_feats_from_server(self, nids):
        response = self.stub.DCSubmit(
            distcache_pb2.DCRequest(
                type=distcache_pb2.get_features_by_client, ids=nids
            ),
            timeout=100000,
        )
        return response.features

    def get_stream_feats_from_server(self, nids):
        err = self.stub_features.DCSubmitFeatures(
            distcache_pb2.DCRequest(
                type=distcache_pb2.get_stream_features_by_client, ids=nids
            ),
            timeout=100000,
        )
        if err != None:
            filename_base = "/dev/shm/repgnn_shm"
            expected_byte_len = len(nids) * self.feat_dim * 4
            expected_feat_chunks = (
                expected_byte_len - 1 + self.feats_chunk_size
            ) // self.feats_chunk_size
            last_ck_size = expected_byte_len % self.feats_chunk_size
            st_ckid, ed_ckid = self.cnt, self.cnt + expected_feat_chunks
            for ckid in range(st_ckid, ed_ckid):
                filename = filename_base + f"{ckid}"
                while True:
                    if os.path.exists(filename):
                        if os.path.getsize(filename) == self.feats_chunk_size:
                            with open(filename, "r+b") as fh:
                                mm = mmap.mmap(fh.fileno(), 0)
                                feats_bytes = bytes(mm.read(self.feats_chunk_size))
                                mm.close()
                            os.remove(filename)
                            self.cnt += 1
                            yield feats_bytes
                            break
                        elif (
                            ckid == ed_ckid - 1
                            and os.path.getsize(filename) == last_ck_size
                        ):
                            with open(filename, "r+b") as fh:
                                mm = mmap.mmap(fh.fileno(), 0)
                                feats_bytes = bytes(mm.read(last_ck_size))
                                mm.close()
                            os.remove(filename)
                            self.cnt += 1
                            yield feats_bytes
                            break
                    else:
                        continue
        else:
            return err

    def get_stream_feats_from_server_v2(self, nids):
        with torch.autograd.profiler.record_function("construct ip2ids"):
            ip2ids = [None] * len(nids)
            curr_idx = 0
            splitlen = []
            nids_np = np.array(nids)
            for i in range(self.world_size):
                tmplst = nids_np[self.Nid2pid[nids_np] == i]
                ip2ids[curr_idx : curr_idx + len(tmplst)] = tmplst.tolist()
                curr_idx += len(tmplst)
                splitlen.append(curr_idx)

        err = self.stub_features.DCSubmitFeatures(
            distcache_pb2.DCRequest(
                type=distcache_pb2.get_stream_features_by_client,
                serids=ip2ids,
                seplen=splitlen,
            ),
            timeout=100000,
        )
        if err != None:
            filename_base = "/dev/shm/repgnn_shm"
            expected_byte_len = len(nids) * self.feat_dim * 4
            expected_feat_chunks = (
                expected_byte_len - 1 + self.feats_chunk_size
            ) // self.feats_chunk_size
            last_ck_size = expected_byte_len % self.feats_chunk_size
            st_ckid, ed_ckid = self.cnt, self.cnt + expected_feat_chunks
            for ckid in range(st_ckid, ed_ckid):
                filename = filename_base + f"{ckid}"
                while True:
                    if os.path.exists(filename):
                        if os.path.getsize(filename) == self.feats_chunk_size:
                            with open(filename, "r+b") as fh:
                                mm = mmap.mmap(fh.fileno(), 0)
                                feats_bytes = bytes(mm.read(self.feats_chunk_size))
                                mm.close()
                            os.remove(filename)
                            self.cnt += 1
                            yield feats_bytes
                            break
                        elif (
                            ckid == ed_ckid - 1
                            and os.path.getsize(filename) == last_ck_size
                        ):
                            with open(filename, "r+b") as fh:
                                mm = mmap.mmap(fh.fileno(), 0)
                                feats_bytes = bytes(mm.read(last_ck_size))
                                mm.close()
                            os.remove(filename)
                            self.cnt += 1
                            yield feats_bytes
                            break
                    else:
                        continue
        else:
            return err

    def get_cache_hit_info(self):
        response = self.stub.DCSubmit(
            distcache_pb2.DCRequest(type=distcache_pb2.get_cache_info), timeout=1000
        )
        return (
            response.requestnum,
            response.localhitnum,
            response.local_feats_gather_time,
            response.remote_feats_gather_time,
        )

    def get_miss_rate(self):
        requestnum, localhitnum, local_feats_gather_time, remote_feats_gather_time = (
            self.get_cache_hit_info()
        )
        self.try_num, self.miss_num = requestnum, requestnum - localhitnum
        miss_rate = float(self.miss_num) / self.try_num
        ret = (self.miss_num, self.try_num, miss_rate)
        self.miss_num = 0
        self.try_num = 0
        return ret

    def get_total_local_remote_feats_gather_time(self):
        requestnum, localhitnum, local_feats_gather_time, remote_feats_gather_time = (
            self.get_cache_hit_info()
        )
        self.local_feats_gather_time = local_feats_gather_time
        self.remote_feats_gather_time = remote_feats_gather_time
        return self.local_feats_gather_time, self.remote_feats_gather_time

    def get_cache_partid(self):
        response = self.stub.DCSubmit(
            distcache_pb2.DCRequest(type=distcache_pb2.get_cache_info), timeout=1000
        )
        return response.partidx

    # help functions
    def merge_lists(self, lists):
        flat_list = np.concatenate(lists)
        unique_values, index = np.unique(flat_list, return_inverse=True)
        return unique_values, index

    def merge_lists_wodedupli(self, lists):
        flat_list = np.concatenate(lists)
        return np.array(flat_list), np.arange(len(flat_list))

    def getMaxNumDivisibleByXYAnd1024(self, x: int, y: int) -> int:
        max_num = 4 * 1024 * 1024  # 4MB
        for i in range(max_num, 0, -1):
            if i % x == 0 and i % y == 0 and i % 1024 == 0:
                return i
        return -1


class DistCacheClientP3:
    def __init__(self, grpc_port, gpu, log, world_size, rank):
        self.grpc_port = grpc_port
        self.channel = grpc.insecure_channel(
            self.grpc_port,
            options=[
                ("grpc.enable_retries", 1),
                ("grpc.keepalive_timeout_ms", 100000),
                ("grpc.max_receive_message_length", 2000 * 1024 * 1024),
            ],
        )
        self.stub = distcache_pb2_grpc.OperatorStub(self.channel)

        self.channel_features = grpc.insecure_channel(
            self.grpc_port,
            options=[
                ("grpc.enable_retries", 1),
                ("grpc.keepalive_timeout_ms", 100000),
                ("grpc.max_receive_message_length", 2000 * 1024 * 1024),
            ],
        )
        self.stub_features = distcache_pb2_grpc.OperatorFeaturesStub(
            self.channel_features
        )

        self.gpuid = gpu
        self.feat_dim = self.get_feat_dim()
        self.dims = {"features": self.feat_dim}
        self.log = log
        self.server_log = self.get_statistic()
        assert self.log == self.server_log, "client and server log flags differ"

        self.try_num = 0
        self.miss_num = 0

        self.cnt = 0

        prefix = "repgnn_shm"
        if any(filename.startswith(prefix) for filename in os.listdir("/dev/shm")):
            os.system("rm " + os.path.join(" /dev/shm", prefix) + "*")

        self.feat_len = (
            self.feat_dim // world_size
            if rank != world_size - 1
            else self.feat_dim // world_size + self.feat_dim % world_size
        )
        self.feats_chunk_size = 4 * 1024 * 1024
        self.feats_chunk_size = self.getMaxNumDivisibleByXYAnd1024(self.feat_len, 4)
        print(f"-> feats chunk size: {self.feats_chunk_size}")

    def fetch_data(self, nodeflow):
        with torch.autograd.profiler.record_function("get nf_nids"):
            nf_nids = nodeflow._node_mapping.tousertensor()
            offsets = nodeflow._layer_offsets
            logging.debug(f"fetch_data batch onid, layer_offset: {nf_nids}, {offsets}")
        logging.debug(f"nf.nlayers: {nodeflow.num_layers}")
        for i in range(nodeflow.num_layers):
            tnid = nf_nids[offsets[i] : offsets[i + 1]]
            with torch.autograd.profiler.record_function("create frames"):
                frame = {
                    name: torch.empty(tnid.size(0), self.dims[name])
                    for name in self.dims
                }
                tnid = tnid.tolist()
            with torch.autograd.profiler.record_function(
                "fetch feat from cache server"
            ):
                features = self.get_feats_from_server(tnid)
            with torch.autograd.profiler.record_function(
                "convert byte features to float tensor"
            ):
                for name in self.dims:
                    frame[name].data = torch.frombuffer(
                        features, dtype=torch.float32
                    ).reshape(len(tnid), self.feat_len)
                    frame[name].data = F.pad(
                        frame[name].data, (0, self.feat_dim - self.feat_len)
                    )
            with torch.autograd.profiler.record_function("move feats from CPU to GPU"):
                for name in self.dims:
                    frame[name].data = frame[name].data.cuda(self.gpuid)
            with torch.autograd.profiler.record_function("asign frame to nodeflow"):
                logging.debug(
                    f'Final nodeflow._node_frames:{i}, frame["features"].size(): {frame["features"].size()}\n'
                )
                if _HAS_DGL:
                    nodeflow._node_frames[i] = FrameRef(Frame(frame))

    def get_feat_dim(self):
        response = self.stub.DCSubmit(
            distcache_pb2.DCRequest(type=distcache_pb2.get_feature_dim), timeout=1000
        )
        return response.featdim

    def Reset(self):
        response = self.stub.DCSubmit(
            distcache_pb2.DCRequest(type=distcache_pb2.reset), timeout=1000
        )

    def get_statistic(self):
        response = self.stub.DCSubmit(
            distcache_pb2.DCRequest(type=distcache_pb2.get_statistic), timeout=1000
        )
        return response.statistic

    def get_feats_from_server(self, nids):
        response = self.stub.DCSubmit(
            distcache_pb2.DCRequest(
                type=distcache_pb2.get_features_by_client, ids=nids
            ),
            timeout=100000,
        )
        return response.features

    def get_cache_hit_info(self):
        response = self.stub.DCSubmit(
            distcache_pb2.DCRequest(type=distcache_pb2.get_cache_info), timeout=1000
        )
        return (
            response.requestnum,
            response.localhitnum,
            response.local_feats_gather_time,
            response.remote_feats_gather_time,
        )

    def get_miss_rate(self):
        requestnum, localhitnum, local_feats_gather_time, remote_feats_gather_time = (
            self.get_cache_hit_info()
        )
        self.try_num, self.miss_num = requestnum, requestnum - localhitnum
        miss_rate = float(self.miss_num) / self.try_num
        ret = (self.miss_num, self.try_num, miss_rate)
        self.miss_num = 0
        self.try_num = 0
        return ret

    def get_total_local_remote_feats_gather_time(self):
        requestnum, localhitnum, local_feats_gather_time, remote_feats_gather_time = (
            self.get_cache_hit_info()
        )
        self.local_feats_gather_time = local_feats_gather_time
        self.remote_feats_gather_time = remote_feats_gather_time
        return self.local_feats_gather_time, self.remote_feats_gather_time

    def get_cache_partid(self):
        response = self.stub.DCSubmit(
            distcache_pb2.DCRequest(type=distcache_pb2.get_cache_info), timeout=1000
        )
        return response.partidx

    def merge_lists(self, lists):
        flat_list = np.concatenate(lists)
        unique_values, index = np.unique(flat_list, return_inverse=True)
        return unique_values, index

    def merge_lists_wodedupli(self, lists):
        flat_list = np.concatenate(lists)
        return np.array(flat_list), np.arange(len(flat_list))

    def getMaxNumDivisibleByXYAnd1024(self, x: int, y: int) -> int:
        max_num = 4 * 1024 * 1024  # 4MB
        for i in range(max_num, 0, -1):
            if i % x == 0 and i % y == 0 and i % 1024 == 0:
                return i
        return -1

    def get_stream_feats_from_server(self, nids):
        err = self.stub_features.DCSubmitFeatures(
            distcache_pb2.DCRequest(
                type=distcache_pb2.get_stream_features_by_client, ids=nids
            ),
            timeout=100000,
        )
        if err != None:
            filename_base = "/dev/shm/repgnn_shm"
            expected_byte_len = len(nids) * self.feat_dim * 4
            expected_feat_chunks = (
                expected_byte_len - 1 + self.feats_chunk_size
            ) // self.feats_chunk_size
            last_ck_size = expected_byte_len % self.feats_chunk_size
            st_ckid, ed_ckid = self.cnt, self.cnt + expected_feat_chunks
            for ckid in range(st_ckid, ed_ckid):
                filename = filename_base + f"{ckid}"
                while True:
                    if os.path.exists(filename):
                        if os.path.getsize(filename) == self.feats_chunk_size:
                            with open(filename, "r+b") as fh:
                                mm = mmap.mmap(fh.fileno(), 0)
                                feats_bytes = bytes(mm.read(self.feats_chunk_size))
                                mm.close()
                            os.remove(filename)
                            self.cnt += 1
                            yield feats_bytes
                            break
                        elif (
                            ckid == ed_ckid - 1
                            and os.path.getsize(filename) == last_ck_size
                        ):
                            with open(filename, "r+b") as fh:
                                mm = mmap.mmap(fh.fileno(), 0)
                                feats_bytes = bytes(mm.read(last_ck_size))
                                mm.close()
                            os.remove(filename)
                            self.cnt += 1
                            yield feats_bytes
                            break
                    else:
                        continue
        else:
            return err
