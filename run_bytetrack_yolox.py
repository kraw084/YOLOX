import torch
import cv2

from exps.custom.yolox_x_ablation import Exp
from tools.demo import Predictor

#python tools/demo.py image -f exps/custom/yolox_x_ablation.py  -c weights/bytetrack_ablation.pth.tar  --path E:\Marine-VOD\MOT17\train\MOT17-02-FRCNN\img1\000001.jpg  --save_result --device [cpu/gpu] --legacy


device = "cpu"

exp = Exp()
exp.test_conf = 0.001
exp.nmsthre = 0.6


model = exp.get_model()
ckpt_file = "YOLOX/weights/bytetrack_ablation.pth.tar"
ckpt = torch.load(ckpt_file, map_location="cpu")
model.load_state_dict(ckpt["model"])

if device == "gpu":
    model.cuda()
model.eval()

yolox_model = Predictor(model, exp, ["Person"], device=device, legacy=True)

im_path = r"E:\Marine-VOD\MOT17\train\MOT17-02-FRCNN\img1\000001.jpg"
output, _ = yolox_model.inference(im_path)

im = cv2.imread(im_path)

output = output[0].cpu().numpy()

im_size_ratio = min(exp.test_size[0] / im.shape[0], exp.test_size[1] / im.shape[1])
output[:, :4] = output[:, :4] / im_size_ratio

formatted_output = []
for bbox in output:
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    conf = bbox[4] * bbox[5]
    label = bbox[6]
    formatted_output.append([x_center, y_center, w, h, conf, label])

    cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 5)
    
cv2.imshow("test", im)
cv2.waitKey(0)



#print(formatted_output)