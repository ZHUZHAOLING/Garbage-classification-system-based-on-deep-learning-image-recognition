# -*- coding: utf-8 -*-
"""
数据集准备脚本
"""
import os
import codecs
import shutil
from tqdm import tqdm
try:
    import moxing as mox
except:
    print('not use moxing')
from glob import glob
from sklearn.model_selection import StratifiedShuffleSplit


def prepare_data_on_modelarts(args):
    """
    如果数据集存储在OBS，则需要将OBS上的数据拷贝到 ModelArts 中
    """
    # Create some local cache directories used for transfer data between local path and OBS path
    if not args.data_url.startswith('s3://'):
        args.data_local = args.data_url
    else:
        args.data_local = os.path.join(args.local_data_root, 'train_val')
        if not os.path.exists(args.data_local):
            mox.file.copy_parallel(args.data_url, args.data_local)
        else:
            print('args.data_local: %s is already exist, skip copy' % args.data_local)

    if not args.train_url.startswith('s3://'):
        args.train_local = args.train_url
    else:
        args.train_local = os.path.join(args.local_data_root, 'model_snapshots')
        if not os.path.exists(args.train_local):
            os.mkdir(args.train_local)

    if not args.test_data_url.startswith('s3://'):
        args.test_data_local = args.test_data_url
    else:
        args.test_data_local = os.path.join(args.local_data_root, 'test_data/')
        if not os.path.exists(args.test_data_local):
            mox.file.copy_parallel(args.test_data_url, args.test_data_local)
        else:
            print('args.test_data_local: %s is already exist, skip copy' % args.test_data_local)

    args.tmp = os.path.join(args.local_data_root, 'tmp')
    if not os.path.exists(args.tmp):
        os.mkdir(args.tmp)

    return args


def split_train_val(input_dir, output_train_dir, output_val_dir):
    """
    大赛发布的公开数据集是所有图片和标签txt都在一个目录中的格式
    如果需要使用 torch.utils.data.DataLoader 来加载数据，则需要将数据的存储格式做如下改变：
    1）划分训练集和验证集，分别存放为 train 和 val 目录；
    2）train 和 val 目录下有按类别存放的子目录，子目录中都是同一个类的图片
    本函数就是实现如上功能，建议先在自己的机器上运行本函数，然后将处理好的数据上传到OBS
    """
    if not os.path.exists(input_dir):
        print(input_dir, 'is not exist')
        return

    # 1. 检查图片和标签的一一对应
    label_file_paths = glob(os.path.join(input_dir, '*.txt'))
    valid_img_names = []
    valid_labels = []
    for file_path in tqdm(label_file_paths):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(',')
        img_name = line_split[0]
        label_id = line_split[1]
        if os.path.exists(os.path.join(input_dir, img_name)):
            valid_img_names.append(img_name)
            valid_labels.append(int(label_id))
        else:
            print('error', img_name, 'is not exist')

    # 2. 使用 StratifiedShuffleSplit 划分训练集和验证集，可保证划分后各类别的占比保持一致
    # TODO，数据集划分方式可根据您的需要自行调整
    sss = StratifiedShuffleSplit(n_splits=1, test_size=3000, random_state=0)
    sps = sss.split(valid_img_names, valid_labels)
    for sp in sps:
        train_index, val_index = sp

    # label_id_name_dict = \
    #     {
    #         "0": "工艺品/仿唐三彩",
    #         "1": "工艺品/仿宋木叶盏",
    #         "2": "工艺品/布贴绣",
    #         "3": "工艺品/景泰蓝",
    #         "4": "工艺品/木马勺脸谱",
    #         "5": "工艺品/柳编",
    #         "6": "工艺品/葡萄花鸟纹银香囊",
    #         "7": "工艺品/西安剪纸",
    #         "8": "工艺品/陕历博唐妞系列",
    #         "9": "景点/关中书院",
    #         "10": "景点/兵马俑",
    #         "11": "景点/南五台",
    #         "12": "景点/大兴善寺",
    #         "13": "景点/大观楼",
    #         "14": "景点/大雁塔",
    #         "15": "景点/小雁塔",
    #         "16": "景点/未央宫城墙遗址",
    #         "17": "景点/水陆庵壁塑",
    #         "18": "景点/汉长安城遗址",
    #         "19": "景点/西安城墙",
    #         "20": "景点/钟楼",
    #         "21": "景点/长安华严寺",
    #         "22": "景点/阿房宫遗址",
    #         "23": "民俗/唢呐",
    #         "24": "民俗/皮影",
    #         "25": "特产/临潼火晶柿子",
    #         "26": "特产/山茱萸",
    #         "27": "特产/玉器",
    #         "28": "特产/阎良甜瓜",
    #         "29": "特产/陕北红小豆",
    #         "30": "特产/高陵冬枣",
    #         "31": "美食/八宝玫瑰镜糕",
    #         "32": "美食/凉皮",
    #         "33": "美食/凉鱼",
    #         "34": "美食/德懋恭水晶饼",
    #         "35": "美食/搅团",
    #         "36": "美食/枸杞炖银耳",
    #         "37": "美食/柿子饼",
    #         "38": "美食/浆水面",
    #         "39": "美食/灌汤包",
    #         "40": "美食/烧肘子",
    #         "41": "美食/石子饼",
    #         "42": "美食/神仙粉",
    #         "43": "美食/粉汤羊血",
    #         "44": "美食/羊肉泡馍",
    #         "45": "美食/肉夹馍",
    #         "46": "美食/荞面饸饹",
    #         "47": "美食/菠菜面",
    #         "48": "美食/蜂蜜凉粽子",
    #         "49": "美食/蜜饯张口酥饺",
    #         "50": "美食/西安油茶",
    #         "51": "美食/贵妃鸡翅",
    #         "52": "美食/醪糟",
    #         "53": "美食/金线油塔"
    #     }
    label_id_name_dict = \
        {
            "0": "一次性快餐盒",
            "1": "污损塑料",
            "2": "烟蒂",
            "3": "牙签",
            "4": "筷子",
            "5": "污损用纸",
            "6": "菜帮菜叶",
            "7": "大骨头",
            "8": "书籍纸张",
            "9": "剩饭剩菜",
            "10": "果皮果肉",
            "11": "鱼骨",
            "12": "茶叶渣",
            "13": "蛋壳",
            "14": "充电宝",
            "15": "包",
            "16": "垃圾桶",
            "17": "塑料器皿",
            "18": "塑料玩具",
            "19": "塑料衣架",
            "20": "快递纸袋",
            "21": "插头电线",
            "22": "旧衣服",
            "23": "易拉罐",
            "24": "枕头",
            "25": "毛绒玩具",
            "26": "洗护用品",
            "27": "玻璃器皿",
            "28": "砧板",
            "29": "纸盒纸箱",
            "30": "花盆",
            "31": "调料瓶",
            "32": "酒瓶",
            "33": "金属厨具",
            "34": "金属器皿",
            "35": "金属食品罐",
            "36": "锅",
            "37": "陶瓷器皿",
            "38": "鞋",
            "39": "食用油桶",
            "40": "饮料瓶",
            "41": "干电池",
            "42": "软膏",
            "43": "过期药物"
    }

    # 3. 创建 output_train_dir 目录下的所有标签名子目录
    for id in label_id_name_dict.keys():
        if not os.path.exists(os.path.join(output_train_dir, id)):
            os.mkdir(os.path.join(output_train_dir, id))

    # 4. 将训练集图片拷贝到 output_train_dir 目录
    for index in tqdm(train_index):
        file_path = label_file_paths[index]
        with codecs.open(file_path, 'r', 'utf-8') as f:
            gt_label = f.readline()
        img_name = gt_label.split('.')[0].strip() + '.jpg'
        id = gt_label.split(',')[1].strip()
        shutil.copy(os.path.join(input_dir, img_name), os.path.join(output_train_dir, id, img_name))
    print("finish trainset!")
    # 5. 创建 output_val_dir 目录下的所有标签名子目录
    for id in label_id_name_dict.keys():
        if not os.path.exists(os.path.join(output_val_dir, id)):
            os.mkdir(os.path.join(output_val_dir, id))

    # 6. 将验证集图片拷贝到 output_val_dir 目录
    for index in tqdm(val_index):
        file_path = label_file_paths[index]
        with codecs.open(file_path, 'r', 'utf-8') as f:
            gt_label = f.readline()
        img_name = gt_label.split('.')[0].strip() + '.jpg'
        id = gt_label.split(',')[1].strip()
        shutil.copy(os.path.join(input_dir, img_name), os.path.join(output_val_dir, id, img_name))
    print("finish valset!")

    print('total samples: %d, train samples: %d, val samples:%d'
          % (len(valid_labels), len(train_index), len(val_index)))
    print('end')


if __name__ == '__main__':
    '''
    total samples: 19414, 
    train samples: 16414, 
    val samples: 3000
    '''

    input_dir = r'F:\dosth\trash_trainval\trainval\new_trainval_from_voc'
    output_train_dir = r'F:\dosth\trash_trainval\trainval\trash_trainval\train'
    output_val_dir = r'F:\dosth\trash_trainval\trainval\trash_trainval\val'
    split_train_val(input_dir, output_train_dir, output_val_dir)
