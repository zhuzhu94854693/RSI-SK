import torch


def class_pro(images1_labeled, images2_labeled):


    average_means = [
        torch.tensor([-0.3994, -0.3631, -0.3794]),
        torch.tensor([-0.2399, -0.2092, -0.2083]),
        torch.tensor([-0.3322, -0.3044, -0.3175]),
        torch.tensor([-0.5621, -0.4798, -0.4829]),
        torch.tensor([-0.4402, -0.4213, -0.4519]),
        torch.tensor([-0.6037, -0.5034, -0.5290]),
        torch.tensor([-0.4034, -0.3599, -0.4183])

    ]

    images_A = images1_labeled.cuda()
    images_B = images2_labeled.cuda()
    size = images_A.shape
    def create_acankao(average_mean):
        acankao = torch.zeros(size).cuda()
        acankao[:, 0, :, :] = average_mean[0]
        acankao[:, 1, :, :] = average_mean[1]
        acankao[:, 2, :, :] = average_mean[2]
        return acankao

    acankaos = [create_acankao(mean) for mean in average_means]

    def calculate_cva(images, acankaos):
        cvas = []
        for acankao in acankaos:
            cva = torch.abs(images[:, 0, :, :] - acankao[:, 0, :, :]) + torch.abs(images[:, 1, :, :] - acankao[:, 1, :, :]) + torch.abs(images[:, 2, :, :] - acankao[:, 2, :, :])

            cvas.append(cva)
        return torch.stack(cvas, dim=1)

    imagA =  calculate_cva(images_A, acankaos)
    classA = torch.argmin(imagA, dim=1)
    classproA = 1- imagA

    imagB = calculate_cva(images_B, acankaos)
    classB = torch.argmin(imagB, dim=1)
    classproB = 1- imagB

  

    return   classA, classB, classproA, classproB,