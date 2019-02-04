import torch


def get_scene(scene, ypred):

    idx = ypred.long()

    return scene[idx[:, 0],
                 idx[:, 1],
                 :]

if __name__=="__main__":
    scene = torch.randn(100, 200, 32)
    ypred = torch.LongTensor(16,2).random_(0, 100)
    print (get_scene(scene, ypred))
