import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from Alexmode import AlexNet
data_tranform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

img = Image.open("./img_2.png")
plt.imshow(img)

img = data_tranform(img)
img = torch.unsqueeze(img,dim=0)

try:
    json_file = open('./class_indices.json','r')
    class_indices = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = AlexNet(num_classes=5)

model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img))
    predict = torch.softmax(output,dim=0)
    predict_cla = torch.argmax(predict).numpy()
    if predict[predict_cla].item()>0.7:
        print(class_indices[str(predict_cla)], predict[predict_cla].item())
    else:
        print("error")
#print(class_indices[str(predict_cla)],predict[predict_cla].item())
plt.show()