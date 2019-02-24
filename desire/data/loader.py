from torch.utils.data import DataLoader

from desire.data.trajectories import TrajectoryDataset, seq_collate


def data_loader(path,
                obs_len=8,
                pred_len=12,
                skip=1,
                threshold=0.002,
                min_ped=1,
                delim='\t',
                batch_size=20,
                loader_num_workers=4):
    dset = TrajectoryDataset(
        path,
        obs_len=obs_len,
        pred_len=pred_len,
        skip=skip,
        delim=delim)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader




# def data_loader(args, path):
#     dset = TrajectoryDataset(
#         path)
#         # obs_len=args.obs_len,
#         # pred_len=args.pred_len,
#         # skip=args.skip,
#         # delim=args.delim)

#     loader = DataLoader(
#         dset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.loader_num_workers,
#         collate_fn=seq_collate)
#     return dset, loader

