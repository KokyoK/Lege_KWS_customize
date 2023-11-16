

############# 打印参数，prepare for C
import sys
import torch
torch.manual_seed(42)
import torch.utils.data as data
import speech_dataset as sd
# import utility as util
import utility_ee as util
import model as md


def format_tensor(tensor, precision=5):
    """递归格式化张量为字符串，保留指定的小数位数。"""
    if tensor.dim() == 0:
        return f'{tensor.item():.{precision}f}'
    elements = [format_tensor(t, precision) for t in tensor]
    return '{' + ', '.join(elements) + '}'
def record_para(model):
    savedStdout = sys.stdout  # 保存标准输出流
    file =  open('paras.txt', 'w+')
    sys.stdout = file  # 标准输出重定向至文件

    s = model.state_dict()
    i = 0
    j = 0
    import torch




    for key in s:
        block_name = key.replace(".","_")
        para = s[key].squeeze()
        # 四舍五入到四位小数
        # para = torch.round(para * 10000) / 10000
        data_str = format_tensor(para)
        # print(block_name)

        if(len(para.shape) == 3):
            print(f"const WDT {block_name}[{para.shape[0]}][{para.shape[1]}][{para.shape[2]}] = {data_str};")
        elif(len(para.shape) == 2):
            print(f"const WDT {block_name}[{para.shape[0]}][{para.shape[1]}] = {data_str};")
        elif(len(para.shape) == 1):
            print(f"const WDT {block_name}[{para.shape[0]}] = {data_str};")


if __name__ == "__main__":
    ROOT_DIR = "dataset/lege/"
    WORD_LIST = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
    SPEAKER_LIST = sd.fetch_speaker_list(ROOT_DIR, WORD_LIST)

    model_fp32 = md.SiameseTCResNet(k=1, n_mels=40, n_classes=len(WORD_LIST), n_speaker=len(SPEAKER_LIST))
    model_fp32.load("sim_244_kwsacc_92.08_idloss_0.0728")
    model_fp32.eval()

    model = model_fp32.network
    model.eval()

    # model.eval()
    record_para(model)
