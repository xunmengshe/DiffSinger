# coding=utf8
print('Starting diffsinger server...')
import json
import os
import pathlib
import sys
import traceback
import utaupy
import zmq

from inference.svs.ds_e2e import DiffSingerE2EInfer
from singer_manager import SingerManager

from typing import List,Tuple

notedict={
    0:"C",
    1:"C#",
    2:"D",
    3:"D#",
    4:"E",
    5:"F",
    6:"F#",
    7:"G",
    8:"G#",
    9:"A",
    10:"A#",
    11:"B"
}

singerManager=SingerManager()

def create_file(filepath:str,mode:str="w",encoding:str="utf-8"):#创建文件夹并新建文本文件
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    return open(filepath,"w")

def poll_socket(socket, timetick = 100):
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    # wait up to 100msec
    try:
        while True:
            obj = dict(poller.poll(timetick))
            if socket in obj and obj[socket] == zmq.POLLIN:
                yield socket.recv()
    except KeyboardInterrupt:
        pass
    # Escape while loop if there's a keyboard interrupt.

def readblocks(ustfile):
    #迭代器，从ust文件中逐个读取块，以字典形式返回
    current_block_data=dict()
    for line in ustfile.readlines():
        if(line.startswith("[")):#块的开始
            #先返回上一个
            if(len(current_block_data)>=1):
                yield current_block_data
            #然后开新块
            current_block_data={"name":line.strip("[#]\n")}
            pass
        else:
            (key,value)=line.strip("\n").split("=")
            current_block_data[key]=value
            pass
    if(len(current_block_data)>=1):#返回最后一个
        yield current_block_data

def writelab(labpath:str,labdata:List[Tuple[float,float,str]]):
    with create_file(labpath,"w",encoding="utf-8") as outputfile:
        for label in labdata:
            outputfile.write("{} {} {}\n".format(
                int(label[0]*10000000),
                int(label[1]*10000000),
                label[2]))

def tick2time(tick:int,tempo:float):
    return tick/(tempo*8)

def timing(input:List[str]):
    ustpath:str=input[0]
    plugin = utaupy.utauplugin.load(ustpath)
    tempo=float(plugin.tempo)
    current_tick=0
    labdata:List[Tuple[float,float,str]]=[]
    for note in plugin.notes:
        note_length=int(note["Length"])
        labdata.append((
            tick2time(current_tick,tempo),
            tick2time(current_tick+note_length,tempo),
            {"R":"pau"}.get(note["Lyric"],note["Lyric"]),
            ))
        current_tick+=note_length
    writelab(ustpath[:-4]+"_enutemp/timing.lab",labdata)
    writelab(ustpath[:-4]+"_enutemp/score.lab",labdata)
    return {
        'path_full_timing': ustpath[:-4]+"_enutemp/timing.full", 
        'path_mono_timing': ustpath[:-4]+"_enutemp/timing.lab"
        }

def vocoder(input:List[str]):
    (ustpath,wavpath)=input
    #解析ust文件为diffsinger所需格式
    #参考：main.py
    plugin = utaupy.utauplugin.load(ustpath)
    tempo:float = float(plugin.tempo)
    singerpath:str = plugin.voicedir
    singer=singerManager.getsinger(singerpath)

    text:List[str]=[]
    ph_seq:List[str]=[]
    note_seq:List[str]=[]
    note_dur_seq:List[str]=[]
    is_slur_seq:List[str]=[]

    for note in plugin.notes:
        lyric=note["Lyric"]
        notenum=int(note["NoteNum"])
        length=int(note["Length"])
        if(lyric=="-"):#；连音符
            ph_seq.append(ph_seq[-1])
            note_seq.append(notedict[notenum%12]+str(notenum//12-1))
            note_dur_seq.append(str(tick2time(length,tempo)))
            is_slur_seq.append("1")
        elif(lyric=="R"):
            text.append("SP")
            ph_seq.append("SP")
            note_seq.append("rest")
            note_dur_seq.append(str(tick2time(length,tempo)))
            is_slur_seq.append("0")
            pass
        else:
            text.append(lyric)
            length_real_time=tick2time(length,tempo)
            note_name=notedict[notenum%12]+str(notenum//12-1)
            for phoneme in singer.phonemeDict[lyric]:
                ph_seq.append(phoneme)
                note_seq.append(note_name)
                note_dur_seq.append(str(length_real_time))
                is_slur_seq.append("0")
    print("Phonemes:"," ".join(ph_seq))
    inp={
        "text":"",
        "ph_seq":" ".join(ph_seq),
        "note_seq":" ".join(note_seq),
        "ph_dur":None,
        "note_dur_seq":" ".join(note_dur_seq),
        "is_slur_seq":" ".join(is_slur_seq),
        'input_type': 'phoneme'
    }
    #将歌手路径以命令行参数的形式传入合成器
    sys.argv = [
        os.path.join(os.path.split(os.path.abspath(sys.argv[0]))[0],"inference/svs/ds_e2e.py"),
        '--config',
        os.path.join(singerpath,"dsconfig.yaml"),
        '--exp_name',
        '0814_opencpop_500k（修复无参音素）'
    ]
    #合成
    DiffSingerE2EInfer.example_run(inp, target=wavpath)
    return {
        'path_wav': wavpath,
    }

#为了方便调试，把argv配置放外面
root_dir = os.path.dirname(__file__)
sys.argv = [
f'{root_dir}/inference/svs/ds_e2e.py',
'--config',
f'{root_dir}/usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml',
'--exp_name',
'0228_opencpop_ds100_rel']

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:15555')
    print('Started diffsinger server')

    for message in poll_socket(socket):
        request = json.loads(message)
        print("="*40)
        print('Received request: %s' % request)
        response = {}
        try:
            request_func_dict={
                "timing":timing,
                'vocoder':vocoder,
            }
            if request[0] in request_func_dict:
                response['result'] = request_func_dict[request[0]](request[1:])
            else:
                raise NotImplementedError('unexpected command %s' % request[0])
        except Exception as e:
            response['error'] = str(e)
            traceback.print_exc()

        print('Sending response: %s' % response)
        socket.send_string(json.dumps(response))

if(__name__=="__main__"):
    main()