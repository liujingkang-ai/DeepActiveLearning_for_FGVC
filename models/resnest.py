import torch
import torch.nn as nn
from resnest.torch import resnest50, resnest101, resnest200


# for pretrained
PATH = {
    'resnest50' : '',
    'resnest101': '',
    'resnest200': '',
    'resnest269': '',
}

in_features = {
    'resnest50' : 2048,
    'resnest101': 2048,
    'resnest200': 2048,
    'resnest269': 2048
}



def net(model_name, pretrained, is_local, change_top=True, NUM_CLASS=1000):

    if model_name == 'resnest50':
        if pretrained and not is_local:
            model = resnest50(pretrained=True)
        else:
            model = resnest50(pretrained=False)
    elif model_name == 'resnest101':
        if pretrained and not is_local:
            model = resnest101(pretrained=True)
        else:
            model = resnest101(pretrained=False)
    elif model_name == 'resnest200':
        if pretrained and not is_local:
            model = resnest200(pretrained=True)
        else:
            model = resnest200(pretrained=False)
    else:
        print('Error model name')

    if is_local:
        model_path = PATH[model_name]
        model.load_state_dict(torch.load(model_path), strict=False)

    in_feature = in_features[model_name]
    if change_top:
        model.fc = nn.Sequential(
            nn.BatchNorm1d(in_feature),
            nn.Linear(in_feature, NUM_CLASS * 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(NUM_CLASS * 4),
            nn.Linear(NUM_CLASS * 4, NUM_CLASS)
        )
    else:
        model.fc = nn.Linear(in_feature, NUM_CLASS)

    return model


if __name__ == "__main__":
    model = net(model_name='resnest50', pretrained=False, is_local=False, change_top=True)
    x = torch.Tensor(3, 3, 224, 224)
    out = model(x)
    print(model)
    print(out.size())