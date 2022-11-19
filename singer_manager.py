import json
import os
from typing import Dict,List
class Singer:
    def __init__(
        self,
        path:str,
        phonemeDict:Dict[str,List[str]],
        dsConfig:dict
        ):
        self.path=path
        self.phonemeDict=phonemeDict
        self.dsConfig=dsConfig
        pass

class SingerManager:
    singerdict:Dict[str,Singer]={}
    def getsinger(self,path:str):
        path=os.path.abspath(path)
        if(not(path in self.singerdict)):
            self.singerdict[path]=loadSinger(path)
        return self.singerdict[path]

def loadSinger(path):
    phonemeDict=json.load(open(os.path.join(path,"dict.json"),encoding="utf8"))
    dsConfig=json.load(open(os.path.join(path,"dsconfig.json"),encoding="utf8"))
    return Singer(path,phonemeDict,dsConfig)