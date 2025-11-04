from plf.utils import WorkFlow
from typing import Dict, Any
import json, os
from copy import deepcopy
from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim



class BasicWorkFlow(WorkFlow):
    def __init__(self):
        super().__init__()
        self.args = {'num_epochs'}

        self.template = {
            'model', 'dataset', 'batch_size'
        }
        self.paths = {"history.history", "quick", 'weights.last'}
        self.logings = {
            "history.history": ['epoch', 'loss']
        }
    
    def _setup(self, args):
        self.num_epochs= args['num_epochs']

    def prepare(self):
        if not self.P.cnfg:
            print("not initiated")
            return
        args = deepcopy(self.P.cnfg["args"])

        self.model = self.load_component(**args["model"])
        ds = self.load_component(**args['dataset'])
        self.trainDataLoader = DataLoader(
            dataset = ds, batch_size = args['batch_size']
        )
        self.resume()
        print("Data loaders are successfully created")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        return True
    
    def resume(self):
        with open(self.P.get_path(of="quick"), encoding="utf-8") as quick:
            quick = json.load(quick)
        self.current_epoch = quick['last']['epoch'] if quick['last']['epoch'] else 0
        
        
    def train_epoch(self, current_epoch):
        total_loss = 0.0
        for x,y in self.trainDataLoader:
            x,y = x.to(self.device), y.to(self.device)
            pred = self.model(x)

            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            total_loss += loss.item()

        self.current_epoch = current_epoch+1
        data = {
                "epoch": current_epoch+1,
                "loss": total_loss/len(self.trainDataLoader)
            }

        self.log(of="history.history",  data = data)

        quick = {
            "last": {
                'epoch':self.current_epoch
                }
            }
        self.log(of="quick", data = quick)
        self.log(of='weights.last')
        print(f"epoch: {self.current_epoch} | Loss: {data['loss']}")

    def run(self):
        for current_epoch in range(self.current_epoch, self.num_epochs):
            if not self.P.should_running:
                return

            self.train_epoch(current_epoch=current_epoch)

    def new(self, args: Dict[str, Any]) -> None:
        if not self.template.issubset(set(args.keys())):
            raise ValueError(f'the args should have {", ".join(self.template- set(list(args.keys())))}')
        
        for i in self.logings:
            record = pd.DataFrame([], columns=self.logings[i])
            record.to_csv(self.P.get_path(of=i), index=False)

        quick = {
            "last": {
                'epoch':0
                }
            }
        self.log(of='quick', data=quick)
        
    def log(self, of, data=None):
        if of=='quick':
            pth = self.P.get_path(of='quick')
            if os.path.exists(pth):
                with open(pth) as fl:
                    qck = json.load(fl)
                qck.update(data)
            else:
                qck = data
            # Write back to the same file
            with open(self.P.get_path(of='quick'), 'w') as fl:
                json.dump(qck, fl, indent=4)

        elif of == 'history.history':
            metrics = self.logings[of]
            record = pd.DataFrame([[data[i] for i in metrics]], columns=metrics)
            record.to_csv(
                self.P.get_path(of=of),
                mode="a",
                header=False,
                index=False,
            )
        elif of == 'weights.last':
            torch.save(
                    self.model.state_dict(),
                    self.P.get_path(of='weights.last'),
                )
    def get_path(self, of, pplid, args= None) -> str:
        if "history.history" == of:
            path = Path(*of.split(".")) / f"{pplid}.csv"
        
        elif of == "quick":
            path = Path("Quicks") / f"{pplid}.json"

        elif of == 'weights.last':
            path = Path("Weights") / "'last" / f"{pplid}.pt"

        else:
            raise ValueError(
                f"Invalid value for 'of': {of}. Supported values: "
                "'config', 'weight', 'gradient', 'history', 'quick'."
            )
        
        return path

    def clean(self):
        pass

    def status(self) -> str:
        with open(self.P.get_path(of='quick')) as fl:
            qck = json.load(fl)
        return qck['last']