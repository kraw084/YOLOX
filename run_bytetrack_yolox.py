import torch

from exps.custom.yolox_x_ablation import Exp
from tools.demo import Predictor

device = "cpu"

exp = Exp()
exp.test_conf = 0.6
exp.nmsthre = 0.6


model = exp.get_model()
ckpt_file = "weights/bytetrack_ablation.pth.tar"
ckpt = torch.load(ckpt_file, map_location="cpu")
model.load_state_dict(ckpt["model"])

if device == "gpu":
    model.cuda()
model.eval()

yolox_model = Predictor(model, exp, ["Person"], device=device, legacy=True)

im_path = r"D:\Marine-VOD\MOT17\train\MOT17-02-FRCNN\img1\000001.jpg"
output, _ = yolox_model.inference(im_path)

output = output[0].cpu().numpy()

formatted_output = []
for bbox in output:
    x_center = bbox[0] + bbox[2] / 2
    y_center = bbox[1] + bbox[3] / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    conf = bbox[4] * bbox[5]
    label = bbox[6]
    formatted_output.append([x_center, y_center, w, h, conf, label])
    print([x_center, y_center, w, h, conf, label])

#print(formatted_output)