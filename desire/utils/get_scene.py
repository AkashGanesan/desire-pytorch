import torch


<<<<<<< HEAD
def get_scene(scene, ypred_rel, x, scene_size):
=======
def get_scene(scene, ypred_rel, x):
>>>>>>> f81a4e8... Fixed get_scene for relative positioning.
    '''get_scene
    input
    =====
    scene: (W/2, H/2, x)  where x == 32
    ypred: (x, y) where x, y are floats
    output:
    '''
    z = ypred_rel + x
    idx = z.long()
<<<<<<< HEAD
    width = scene_size[0]
    height = scene_size[1]
    shrinkage = scene_size[2]
    x = idx[:, 0]
    y = idx[:, 1]
    x = width // 2 + x
    y = height // 2 - y
    x = x.long()
    y = y.long()
    x = torch.clamp(x, 0, width)
    y = torch.clamp(y, 0, height)
    # print("Dims of scene and indices",
    #       (scene.size,
    #       idx[:, 0] // 2,
    #       idx[:, 1] // 2,))
    return scene[:,
                 x // shrinkage,
                 y // shrinkage].transpose(0, 1)


if __name__ == "__main__":
    scene_size = (100, 200, 2)
    scene = torch.randn(32, 100, 200)
    x_start = torch.randn(16,2)
    ypred = torch.randn(16,2)
    a = get_scene(scene, ypred, x_start, scene_size)
=======

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

>>>>>>> f81a4e8... Fixed get_scene for relative positioning.
