import torch


def class_pro(images1_labeled, images2_labeled):


    average_means = [
        torch.tensor([-0.3318, -0.3626, -0.3751]),
        torch.tensor([ 0.1396,  0.0957, -0.0542]),
        torch.tensor([-0.2489, -0.2797, -0.3881]),
        torch.tensor([-0.5058, -0.4813, -0.5539]),
        torch.tensor([-0.3981, -0.4399, -0.5341]),
        torch.tensor([-0.6469, -0.6657, -0.7212]),
        torch.tensor([-0.3202, -0.3496, -0.4735])
    ]

    average_means_2 = [
        torch.tensor([-0.4958, -0.4811, -0.5679]),
        torch.tensor([-0.4030, -0.3952, -0.4998]),
        torch.tensor([-0.4551, -0.4450, -0.5423]),
        torch.tensor([-0.4920, -0.4764, -0.5665]),
        torch.tensor([-0.4687, -0.4582, -0.5540]),
        torch.tensor([-0.4960, -0.4847, -0.5743]),
        torch.tensor([-0.4569, -0.4467, -0.5458])
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
    acankaos_2 = [create_acankao(mean) for mean in average_means_2]

    def calculate_cva(images, acankaos):
        cvas = []
        for acankao in acankaos:
            cva = torch.abs(images[:, 0, :, :] - acankao[:, 0, :, :]) +torch.abs(images[:, 1, :, :] - acankao[:, 1, :, :]) +torch.abs(images[:, 2, :, :] - acankao[:, 2, :, :])

            cvas.append(cva)
        return torch.stack(cvas, dim=1)

    imagA =  calculate_cva(images_A, acankaos)
    classA = torch.argmin(imagA, dim=1)
    classproA = (1- imagA)

    imagB = calculate_cva(images_B, acankaos_2)
    classB = torch.argmin(imagB, dim=1)
    classproB = (1- imagB)

    imagC = calculate_cva(images_B, acankaos)
    classC = torch.argmin(imagC, dim=1)
    classproC = (1 - imagC)

    imagD = calculate_cva(images_A, acankaos_2)
    classD = torch.argmin(imagD, dim=1)
    classproD = (1 - imagD)



    diff_means = [average_means_2[i] - average_means[i] for i in range(len(average_means))]

    result_tensor = torch.zeros_like(images_A).cuda()


    for i in range(len(diff_means)):
        class_mask = (classA == i)
        result_tensor[:, 0, :, :][class_mask] = diff_means[i][0]
        result_tensor[:, 1, :, :][class_mask] = diff_means[i][1]
        result_tensor[:, 2, :, :][class_mask] = diff_means[i][2]

    result_tensor_chayi = result_tensor + images_A

    imagE = calculate_cva(result_tensor_chayi, acankaos_2)
    classE = torch.argmin(imagE, dim=1)
    classproE = (1 - imagE)



    result_tensor_2 = torch.zeros_like(images_B).cuda()


    for i in range(len(diff_means)):
        class_mask_2 = (classB == i) 
        result_tensor_2[:, 0, :, :][class_mask_2] = diff_means[i][0]
        result_tensor_2[:, 1, :, :][class_mask_2] = diff_means[i][1]
        result_tensor_2[:, 2, :, :][class_mask_2] = diff_means[i][2]

    result_tensor_chayi_2 =  images_B - result_tensor

    imagF = calculate_cva(result_tensor_chayi_2, acankaos)
    classF = torch.argmin(imagF, dim=1)
    classproF = (1 - imagF)

    return  classA, classB, classproA, classproB, classproC, classproD, classproE, classproF