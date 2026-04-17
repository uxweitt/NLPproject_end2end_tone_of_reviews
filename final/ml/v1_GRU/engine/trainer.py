import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import datetime as dt
import numpy as np

from data_set.data_set import ReviewDataset, collate_fn

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path


class Trainer():
    """
    Parameters:
        dataset: пользовательский класс, предобрабатывающий данные
        loss_f: функция потерь
        learning_rate: величина градиентного шага
        epoch_amount: общее количество эпох
        batch_size: размер одного бача
        device: устройство для вычислений
        early_stopping: количество эпох без улучшений до остановки обучения
        optim: оптимизатор
        scheduler: регулятор градиентного шага
        permutate: перемешивание тренировочной выборки перед обучением
        save_intermediate: сохранять ли чекпоинты модели во время обучения
        save_best: сохранять ли лучшую модель

    Attributes:
        start_model: необученная модель
        best_model: модель, после обучения
        train_loss: средние значения функции потерь на тренировочных 
                    данных в каждой эпохе
        val_loss: средние значения функции потерь на валидационных 
                  данных в каждой эпохе

    Methods:
        fit: обучение модели
        predict: возвращает предсказание обученной моделью

    """
    def __init__(self,  dataset, net, loss_f, learning_rate=1e-3, 
                epoch_amount=10, batch_size=12, 
                max_batches_per_epoch=None,
                device='cpu', early_stopping=10, 
                optim=torch.optim.Adam, 
                scheduler=None, permutate=True,
                save_best=True, save_intermediate=False):
        
        self.loss_f = loss_f
        self.learning_rate = learning_rate
        self.epoch_amount = epoch_amount
        self.batch_size = batch_size
        self.max_batches_per_epoch = max_batches_per_epoch
        self.device = device
        self.early_stopping = early_stopping
        self.optim = optim
        self.scheduler = scheduler
        self.permutate = permutate
        self.dataset = dataset
        self.start_model = net
        self.best_model = net
        self.save_best = save_best
        self.save_intermediate = save_intermediate

        self.train_loss = []
        self.val_loss = []

    def predict(self, X):
        return self.best_model(X)

    def fit(self):
        Net = self.start_model
        Net.to(self.device)
        optimizer = self.optim(Net.parameters(), lr=self.learning_rate)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)

        train, val = data.random_split(self.dataset, [0.8, 0.2])
        train = data.DataLoader(train, 
                                batch_size=self.batch_size, 
                                shuffle=self.permutate,
                                collate_fn=collate_fn) 
        val = data.DataLoader(val, 
                                batch_size=self.batch_size, 
                                shuffle=False,
                                collate_fn=collate_fn)

        best_val_loss = float('inf') # Лучшее значение функции потерь на валидационной выборке
        best_ep = 0                  # Эпоха, на которой достигалось лучшее значение функции потерь на валидационной выборке

        for epoch in range(self.epoch_amount): 
            start = dt.datetime.now()
            Net.train()
            mean_loss = 0
            batch_n = 0

            train_tqdm = tqdm(train, leave=True)
            for batch_X, target in train_tqdm:
                optimizer.zero_grad()

                batch_X = batch_X.to(self.device)
                target = target.to(self.device)

                predicted_values = Net(batch_X).squeeze(0)
                loss = self.loss_f(predicted_values, target)
                loss.backward()
                optimizer.step()

                batch_n += 1
                mean_loss = 1/batch_n*loss.item() + (1-1/batch_n)*mean_loss
                train_tqdm.set_description(
                f"Epoch [{epoch+1}/{self.epoch_amount}], loss_train={mean_loss:.3f}, {dt.datetime.now() - start} сек"
                )

        
            mean_loss /= batch_n
            self.train_loss.append(mean_loss)

            Net.eval()
            mean_loss = 0
            batch_n = 0

            with torch.no_grad():
                val_tqdm = tqdm(val, leave=True)
                for batch_X, target in val_tqdm:
                    batch_X = batch_X.to(self.device)
                    target = target.to(self.device)

                    predicted_values = Net(batch_X)
                    loss = self.loss_f(predicted_values, target)

                    batch_n += 1
                    mean_loss = 1/batch_n*loss.item() + (1-1/batch_n)*mean_loss
                    val_tqdm.set_description(
                    f"Epoch [{epoch+1}/{self.epoch_amount}], loss_val={mean_loss:.3f}, {dt.datetime.now() - start} сек"
                    )
            
            mean_loss /= batch_n
            self.val_loss.append(mean_loss)

            if mean_loss < best_val_loss:
                self.best_model = Net
                best_val_loss = mean_loss
                best_ep = epoch
                if self.save_intermediate:
                    self.save_model(name_models=
                    f"Epoch_{epoch+1}_loss_val_{best_val_loss:.3f}.pt"
                    )
                
            elif epoch - best_ep > self.early_stopping:
                print(f'{self.early_stopping} без улучшений. Прекращаем обучение...')
                break
            if self.scheduler is not None:
                scheduler.step()
                
        if self.save_best:
            self.save_model(name_models=f"best_loss_val_{best_val_loss:.3f}.pt")
    
    def save_model(self, save_dir = "models", name_models='best.pt'):
        BASE_DIR = Path(__file__).resolve().parent.parent
        SAVE_DIR = save_dir
        torch.save(self.best_model.state_dict(),
                    BASE_DIR / SAVE_DIR / name_models)
        