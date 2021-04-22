import torch
from skimage import measure


def osnet(fms):
    A = torch.sum(fms, dim=1, keepdim=True)
    # print('A.shape:', A.shape)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    # print('a.shape:', a.shape)
    M = (A > a).float()
    # print(M.shape)

#     A1 = torch.sum(fm1, dim=1, keepdim=True)
#     a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
#     M1 = (A1 > a1).float()

    coordinates = []
    # torch.Size([8, 1, 14, 14])
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(16, 16)
        # array(14, 14)
        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        # print(len(areas))
        max_idx = areas.index(max(areas))


        intersection = (component_labels==(max_idx+1)).astype(int) == 1
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, 16, 16]
            print('there is one img no intersection')
        else:
            bbox = prop[0].bbox


        x_lefttop = bbox[0] * 16 - 1
        y_lefttop = bbox[1] * 16 - 1
        x_rightlow = bbox[2] * 16 - 1
        y_rightlow = bbox[3] * 16 - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    return coordinates

