import os, pytorch_lightning as pl, torch, wandb, pandas as pd, cv2, torchmetrics, torchvision, timm
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split, DataLoader
from focal_loss import BinaryFocalLoss

class MIPFold(torch.utils.data.Dataset):
    def __init__(self, set: str, fold: int, dir: str = "", csv_file: str = "dataset.csv",
    target: str = "binary_rankin", augmentations: list = [], in_channels: int = 1):
        assert set in ("train", "val", "test")
        assert 0 <= fold < 5
        if set == "test": assert augmentations == []
        augmentations = ["flip", "elastic_deformation", "flip_elastic_deformation"] if augmentations == "all" else augmentations
        csv_file      = os.path.join(dir, csv_file)
        csv_file      = pd.read_csv(csv_file)
        csv_file      = csv_file[csv_file[f"fold{fold+1}"] == set]
        dataset       = {"patient_id":[], "augmentation":[], "target":[]}
        for _, row in csv_file.iterrows():
            dataset["patient_id"]   .append(row.patient_id)
            dataset["target"]       .append(row[target])
            dataset["augmentation"] .append(None)
            for augmentation in augmentations:
                dataset["patient_id"]   .append(row.patient_id)
                dataset["target"]       .append(row[target])
                dataset["augmentation"] .append(augmentation)
        self.dataset = pd.DataFrame(dataset)
        self.dir     = dir
        self.in_channels = in_channels
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        patient_id   = self.dataset.iloc[i].patient_id
        label        = self.dataset.iloc[i].target
        augmentation = self.dataset.iloc[i].augmentation
        if augmentation is not None:
            patient_id = f"{patient_id}-{augmentation}"
        if self.in_channels == 1:
            mip = cv2.imread(os.path.join(self.dir, f"{patient_id}.png"), cv2.IMREAD_GRAYSCALE)
            mip = np.expand_dims(mip, axis = 0)
        else:
            mip = cv2.imread(os.path.join(self.dir, f"{patient_id}.png"), cv2.IMREAD_COLOR).transpose(2,0,1)
        return mip, label
        
        
class MIPFoldDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, csv_file: str, fold: int, dir: str = "", 
    target: str = "binary_rankin", augmentations: list = []):
        super().__init__()
        self.batch_size  = batch_size
        self.csv_file    = csv_file
        self.fold        = fold
        self.kwargs      = {"dir": dir, "target": target, "augmentations": augmentations, "csv_file": csv_file, "in_channels": 3}
        self.cuda        = torch.cuda.is_available()
    def prepare_data(self):
        pass
    def setup(self, stage = None):
        self.train = MIPFold("train", self.fold, **self.kwargs)
        self.val   = MIPFold("val",   self.fold, **self.kwargs)
        # del self.kwargs["augmentations"]
        # self.test  = MIPFold("test",  self.fold, **self.kwargs)
        self.train.dataset.transform = None
        self.val.dataset.transform   = None
        # self.test.dataset.transform  = None
    def train_dataloader(self):
        return DataLoader(self.train, batch_size = self.batch_size, shuffle = True, num_workers = 8 if self.cuda else 1, pin_memory = self.cuda)
    def val_dataloader(self):
        return DataLoader(self.val, batch_size = self.batch_size, num_workers = 8 if self.cuda else 1, pin_memory = self.cuda)
    def test_dataloader(self):
        return DataLoader(self.val, batch_size = self.batch_size, num_workers = 8 if self.cuda else 1, pin_memory = self.cuda)

class Siamese(torch.nn.Module):
    def __init__(self, in_channels = 1, transfer: bool = False):
        super().__init__()
        self.encoder = timm.create_model(NET, pretrained = True)
        if transfer:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        if "resnet" in NET:
            self.encoder.fc = torch.nn.Identity()
        else:
            self.encoder.classifier = torch.nn.Identity()
        # self.encoder.layer4 = torch.nn.Identity()
        self.encoder.global_pool = torch.nn.Identity()
        self.pooling = torch.nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.encoder(x.flip(3))
        x = torch.abs(x1 - x2)
        x = self.pooling(x).squeeze()
        return x

class LitModel(pl.LightningModule):
    # adapted from:
    # https://colab.research.google.com/drive/1smfCw-quyKwxlj69bbsqpZhD75CnBuRh?usp=sharing#scrollTo=qu0xf25aUckF
    def __init__(self, input_shape, num_classes, learning_rate = 2e-4, weight_decay = 0, transfer = False):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate     = learning_rate
        self.weight_decay      = weight_decay
        self.dim               = input_shape
        self.num_classes       = num_classes
        # self.feature_extractor = torchvision.models.resnet50(pretrained = transfer)
        # self.feature_extractor.fc = torch.nn.Identity()
        # self.feature_extractor.classifier = torch.nn.Identity()
        # if transfer:
        #     self.feature_extractor.eval()
        #     for param in self.feature_extractor.parameters():
        #         param.requires_grad = False
        self.feature_extractor = Siamese(in_channels = 3, transfer = transfer)
        # n_sizes         = self._get_conv_output(input_shape)
        # self.classifier = torch.nn.Sequential(torch.nn.Linear(256, 128),
        #                                         torch.nn.Linear(128, self.num_classes))
        self.classifier = torch.nn.Linear(1280,self.num_classes)
        # self.criterion  = BinaryFocalLoss(alpha = .25, gamma = 2, reduction = "mean")
        self.criterion  = torch.nn.BCELoss(reduction = "mean")
        self.accuracy   = torchmetrics.Accuracy()
        self.auc        = torchmetrics.AUROC(num_classes = None, pos_label = 1)
        self.f1_score   = torchmetrics.F1Score()
    def _get_conv_output(self, shape):
        '''
        returns the size of the output tensor going into the Linear layer from the conv block.
        '''
        batch_size  = 1
        tmp_input   = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.feature_extractor(tmp_input) 
        n_size      = output_feat.data.view(batch_size, -1).size(1)
        return n_size
    def forward(self, x):
        # x1, x2    = x[:,:,:,:x.shape[3]//2], x[:,:,:,x.shape[3]//2:].flip(3)
        # x = self.feature_extractor(x1, x2)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.num_classes == 1:
            x = torch.sigmoid(x)
        return x.squeeze()
    def training_step(self, batch):
        batch, gt = batch[0].float(), batch[1]
        out  = self.forward(batch)
        loss = self.criterion(out, gt.type(torch.float32))
        acc  = self.accuracy (out, gt)
        auc  = self.auc      (out, gt)
        f1   = self.f1_score (out, gt)
        self.log("train/loss", loss, on_epoch = True, on_step = False)
        self.log("train/acc", acc, on_epoch = True, on_step = False)
        self.log("train/auc", auc, on_epoch = True, on_step = False)
        self.log("train/f1", f1, on_epoch = True, on_step = False)
        return loss
    def validation_step(self, batch, batch_idx):
        batch, gt = batch[0].float(), batch[1]
        out  = self.forward(batch)
        loss = self.criterion(out, gt.type(torch.float32))
        acc  = self.accuracy (out, gt)
        auc  = self.auc      (out, gt)
        f1   = self.f1_score (out, gt)
        self.log("val/loss", loss, on_epoch = True, on_step = False)
        self.log("val/acc", acc, on_epoch = True, on_step = False)
        self.log("val/auc", auc, on_epoch = True, on_step = False)
        self.log("val/f1", f1, on_epoch = True, on_step = False)
        return loss
    def test_step(self, batch, batch_idx):
        batch, gt = batch[0].float(), batch[1]
        out  = self.forward(batch)
        loss = self.criterion(out, gt.type(torch.float32))
        return {"loss": loss, "outputs": out, "gt": gt}
    def test_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        out  = torch.cat([x["outputs"] for x in outputs], dim = 0)
        gts  = torch.cat([x["gt"] for x in outputs], dim = 0)
        acc  = self.accuracy(out, gts)
        auc  = self.auc     (out, gts)
        f1   = self.f1_score(out, gts)
        self.log("test/loss", loss)
        self.log("test/acc", acc)
        self.log("test/auc", auc)
        self.log("test/f1", f1)
        self.log("lr", self.scheduler.get_lr()[0])
        self.test_gts    = gts
        self.test_output = out
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)
        return [optimizer], [self.scheduler]
        
if __name__ == "__main__":
    wandb.login()
    target  = "occlusion"
    INESC   = torch.cuda.is_available()
    DIR     = "/media/avcstorage/gravo/MIP" if INESC else "../../data/MIP"
    NET     = "efficientnet_b0"
    name    = f"Siamese-{NET}-{target}"
    with open(f"{name}.csv", "w+") as f:
        f.write("fold,auc,f1,accuracy\n")
        for i in range(5):
            checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor = "val/loss", mode = "min")
            trainer = pl.Trainer(logger = WandbLogger(project = "mip4", name = f"{name}-fold{i+1}"), 
                                max_epochs = 150, 
                                accelerator = "gpu" if INESC else "cpu",
                                log_every_n_steps = 20,
                                callbacks = [checkpoint_callback])
            fold  = MIPFoldDataModule(32, f"dataset-{target}.csv", i, DIR, augmentations = [], target = target)
            model = LitModel(input_shape  = (3, 218, 182),
                            num_classes   = 1,
                            learning_rate = .0002,
                            weight_decay  = .001,
                            transfer      = True)
            trainer.fit(model, fold)
            performance = trainer.test(model, datamodule = fold, ckpt_path = "best")[0]
            f.write(f"{i+1},{performance['test/auc']},{performance['test/f1']},{performance['test/acc']}\n")
            wandb.finish()
