import os
import codecs
from PIL import Image
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
from efficientnet_pytorch import EfficientNet
import moxing as mox
import argparse


parser = argparse.ArgumentParser(description='model eval in ModelArts')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b7',
                    help='model architecture')
parser.add_argument('--model_path',
                    default='s3://hwyb-contest/model_snapshots_bp/V0020/model/model_best.pth',
                    type=str, help='path for eval model')
parser.add_argument('--image_path', type=str,
                    default='s3://hwyb-contest/dataset/train_data_source/',
                    help='image folder that need to be eval. labels should be included')
parser.add_argument('--train_url', required=True, type=str, help='the path to save training outputs')

args, unknown = parser.parse_known_args()


class ImageClassificationService():
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model_name_sub = args.arch
        self.model = EfficientNet.from_name(self.model_name_sub)
        feature = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features=feature, out_features=54, bias=True)
        self.use_cuda = False
        if torch.cuda.is_available():
            print('Using GPU for inference')
            self.use_cuda = True
            checkpoint = torch.load(self.model_path)
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            print('Using CPU for inference')
            checkpoint = torch.load(self.model_path, map_location='cpu')
            state_dict = OrderedDict()
            # 训练脚本 main.py 中保存了'epoch', 'arch', 'state_dict', 'best_acc1', 'optimizer'五个key值，
            # 其中'state_dict'对应的value才是模型的参数。
            # 训练脚本 main.py 中创建模型时用了torch.nn.DataParallel，因此模型保存时的dict都会有‘module.’的前缀，
            # 下面 tmp = key[7:] 这行代码的作用就是去掉‘module.’前缀
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                state_dict[tmp] = value
            self.model.load_state_dict(state_dict)

        self.model.eval()

        self.idx_to_class = checkpoint['idx_to_class']
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.image_size = EfficientNet.get_image_size(self.model_name_sub)
        self.transforms = transforms.Compose([transforms.Resize(self.image_size),
                                              transforms.CenterCrop(self.image_size),
                                              transforms.ToTensor(),
                                              self.normalize]
                                             )

        self.label_id_name_dict = \
            {
                "0": "工艺品/仿唐三彩",
                "1": "工艺品/仿宋木叶盏",
                "2": "工艺品/布贴绣",
                "3": "工艺品/景泰蓝",
                "4": "工艺品/木马勺脸谱",
                "5": "工艺品/柳编",
                "6": "工艺品/葡萄花鸟纹银香囊",
                "7": "工艺品/西安剪纸",
                "8": "工艺品/陕历博唐妞系列",
                "9": "景点/关中书院",
                "10": "景点/兵马俑",
                "11": "景点/南五台",
                "12": "景点/大兴善寺",
                "13": "景点/大观楼",
                "14": "景点/大雁塔",
                "15": "景点/小雁塔",
                "16": "景点/未央宫城墙遗址",
                "17": "景点/水陆庵壁塑",
                "18": "景点/汉长安城遗址",
                "19": "景点/西安城墙",
                "20": "景点/钟楼",
                "21": "景点/长安华严寺",
                "22": "景点/阿房宫遗址",
                "23": "民俗/唢呐",
                "24": "民俗/皮影",
                "25": "特产/临潼火晶柿子",
                "26": "特产/山茱萸",
                "27": "特产/玉器",
                "28": "特产/阎良甜瓜",
                "29": "特产/陕北红小豆",
                "30": "特产/高陵冬枣",
                "31": "美食/八宝玫瑰镜糕",
                "32": "美食/凉皮",
                "33": "美食/凉鱼",
                "34": "美食/德懋恭水晶饼",
                "35": "美食/搅团",
                "36": "美食/枸杞炖银耳",
                "37": "美食/柿子饼",
                "38": "美食/浆水面",
                "39": "美食/灌汤包",
                "40": "美食/烧肘子",
                "41": "美食/石子饼",
                "42": "美食/神仙粉",
                "43": "美食/粉汤羊血",
                "44": "美食/羊肉泡馍",
                "45": "美食/肉夹馍",
                "46": "美食/荞面饸饹",
                "47": "美食/菠菜面",
                "48": "美食/蜂蜜凉粽子",
                "49": "美食/蜜饯张口酥饺",
                "50": "美食/西安油茶",
                "51": "美食/贵妃鸡翅",
                "52": "美食/醪糟",
                "53": "美食/金线油塔"
            }

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = self.transforms(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        img = data["input_img"]
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred_score = self.model(img)
            pred_score = F.softmax(pred_score.data, dim=1)
            if pred_score is not None:
                pred_label = torch.argsort(pred_score[0], descending=True)[:1][0].item()
                pred_label = self.idx_to_class[int(pred_label)]
                result = {'result': self.label_id_name_dict[str(pred_label)]}
            else:
                result = {'result': 'predict score is None'}

        return result

    def _postprocess(self, data):
        return data


def infer_on_dataset(img_dir, label_dir, model_path):
    start_all = time.time()
    if not os.path.exists(img_dir):
        print('img_dir: %s is not exist' % img_dir)
        return None
    if not os.path.exists(label_dir):
        print('label_dir: %s is not exist' % label_dir)
        return None
    if not os.path.exists(model_path):
        print('model_path: %s is not exist' % model_path)
        return None
    output_dir = model_path + '_output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    infer = ImageClassificationService('', model_path)
    files = os.listdir(img_dir)
    error_results = []
    right_count = 0
    total_count = 0
    for file_name in files:
        start = time.time()
        if not file_name.endswith('jpg'):
            continue

        with codecs.open(os.path.join(label_dir, file_name.split('.jpg')[0] + '.txt'), 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_name.split('.jpg')[0] + '.txt'))
            continue
        gt_label = infer.label_id_name_dict[line_split[1]]
        # gt_label = "工艺品/仿唐三彩"

        img_path = os.path.join(img_dir, file_name)
        print(img_path)
        img = Image.open(img_path)
        img = infer.transforms(img)
        result = infer._inference({"input_img": img})
        pred_label = result.get('result', 'error')

        total_count += 1
        if pred_label == gt_label:
            right_count += 1
        else:
            error_results.append(', '.join([file_name, gt_label, pred_label]) + '\n')
        print(file_name, ' : ', time.time()-start)

    acc = float(right_count) / total_count
    result_file_path = os.path.join(output_dir, 'accuracy.txt')
    with codecs.open(result_file_path, 'w', 'utf-8') as f:
        f.write('# predict error files\n')
        f.write('####################################\n')
        f.write('file_name, gt_label, pred_label\n')
        f.writelines(error_results)
        f.write('####################################\n')
        f.write('accuracy: %s\n' % acc)
    print('accuracy result file saved as %s' % result_file_path)
    print('accuracy: %0.4f' % acc)
    print('total time:', time.time()-start_all)
    return output_dir


def files_copy(source_path, target_path):
    # transfer files between S3 and Docker
    # all files in source_path will be copied to target_path
    print('-----copying files-----')
    print('From: ', source_path)
    print('To:   ', target_path)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    mox.file.copy_parallel(source_path, target_path)
    print('------copy over------')


if __name__ == '__main__':
    img_dir = '/cache/image/'
    label_dir = '/cache/image/'
    model_path = '/cache/model/model_best.pth'
    result_path = args.train_url
    files_copy(args.image_path, img_dir)
    mox.file.copy(args.model_path, model_path)
    output = infer_on_dataset(img_dir, label_dir, model_path)
    files_copy(output, result_path)
