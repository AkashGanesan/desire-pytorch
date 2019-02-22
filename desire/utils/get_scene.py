import torch


def get_scene(scene, ypred_rel, x):
    '''get_scene
    input
    =====
    scene: (W/2, H/2, x)  where x == 32
    ypred: (x, y) where x, y are floats
    output:
    '''
    z = ypred_rel + x
    idx = z.long()

    # print("Dims of scene and indices",
    #       (scene.size,
    #       idx[:, 0] // 2,
    #       idx[:, 1] // 2,))
    return scene[:,
                 idx[:, 0] // 2,
                 idx[:, 1] // 2].transpose(0,1)

if __name__=="__main__":
    scene = torch.randn(100, 200, 32)
    x_start = torch.randn(16,2)
    ypred = torch.randn(16,2)
    print (get_scene(scene, ypred, x_start))

