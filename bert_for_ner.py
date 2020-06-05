"""
bert for ner
@author:zuolong
@time:2020.06.05
"""

"""
1.超参
2.加载数据，数据处理
3.建模
4.数据生成器
"""

from bert4keras.tokenizers import Tokenizer,load_vocab
from bert4keras.models import build_transformer_model,Model
from bert4keras.layers import Dense,Dropout,ConditionalRandomField
from bert4keras.optimizers import Adam
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping
from keras.utils import to_categorical
import argparse
import os,sys
import numpy as np

parse = argparse.ArgumentParser()
parse.add_argument('-d','--TRAIN_DATA_PATH',default=os.path.join(sys.path[0],'train.txt'),help='训练数据路径')
parse.add_argument('-t','--TEST_DATA_PATH',default=os.path.join(sys.path[0],'test.txt'),help='测试数据路径')
parse.add_argument('-c','--BERT_CONFIG',default=os.path.join(os.path.dirname(os.path.dirname(sys.path[0])),'chinese_L-12_H-768_A-12','bert_config.json'),help='bert配置路径')
parse.add_argument('-m','--BERT_MODEL',default=os.path.join(os.path.dirname(os.path.dirname(sys.path[0])),'chinese_L-12_H-768_A-12','bert_model.ckpt'),help='bert模型路径')
parse.add_argument('-v','--BERT_VOCAB',default=os.path.join(os.path.dirname(os.path.dirname(sys.path[0])),'chinese_L-12_H-768_A-12','vocab.txt'),help='bert模型词汇表')
parse.add_argument('-ck','--MODEL_PATH',default=os.path.join(os.path.dirname(os.path.dirname(sys.path[0])),'checkpoints','model.h5'),help='模型保存路径')
parse.add_argument('-l','--BERT_LAYER',default='Transformer-11-FeedForward-Norm',help='bert模型修改层')
parse.add_argument('-b','--BATCH_SIZE',default=32,help='batch size')
parse.add_argument('-e','--EPOCHS',default=2,help='epochs')
parse.add_argument('-M','--MAX_LEN',default=68,help='文本句长')
args = parse.parse_args()

_labels = ['TIME','LOC','PER','ORG']
_labels_num = len(_labels)*2 + 1

# 加载分词器
token_dict = load_vocab(dict_path=args.BERT_VOCAB)
tokenizer = Tokenizer(token_dict=token_dict)
token_head = tokenizer._token_start_id
token_end = tokenizer._token_end_id

def id_label_dict():
    """
    标注与数字映射词典
    :return:
    """
    id2label = dict(enumerate(_labels))
    label2id = {}
    for k,v in id2label.items():
        label2id[v] = k
    return label2id,id2label

# 加载字典
label2id, id2label = id_label_dict()
print(label2id, id2label)

class DataGenerator():
    def __init__(self,data,batchsize = args.BATCH_SIZE):
        self.data = data
        self.batchsize = batchsize
        self.steps = len(self.data) // self.batchsize
        if len(self.data) % self.batchsize != 0 :
            self.steps += 1

    def __iter__(self,):
        while True:
            X1,X2,Y = [],[],[]
            idx = list(range(len(self.data)))
            np.random.shuffle(idx)
            for i in idx:
                _data = self.data[i]
                x1,x2,y = [],[],[]
                for _t in _data:
                    text = _t[0]
                    label = _t[1].strip()
                    token1,token2 = tokenizer.encode(first_text=text)
                    x1 += token1[1:-1]
                    x2 += token2[1:-1]
                    if label == 'O':
                        y += [0]*len(token1[1:-1])
                    elif label != 'O':
                        # print("label:",label)
                        B_head = label2id[label]*2+1
                        I_back = label2id[label]*2+2
                        y += [B_head]+[I_back]*(len(token1[1:-1])-1)
                x1 = [token_head] + x1 + [token_end]
                x2 = [0] + x2 + [0]
                y = [0] + y + [0]
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1)==self.batchsize or i == idx[-1]:
                    X1 = self.data_padding(data=X1)
                    X2 = self.data_padding(data=X2)
                    Y = self.data_padding(data=Y)
                    # Y = to_categorical(Y,num_classes=_labels_num)
                    yield [X1,X2],Y
                    X1,X2,Y = [],[],[]

    def data_padding(self,data,padding=0):
        """
        data padding
        :param data:
        :param padding:
        :return:
        """
        data_len = [len(d) for d in data]
        M_L = max(data_len)
        return np.array(
            [np.concatenate([d,[padding]*(M_L-len(d))]) if len(d)< M_L else d for d in data]
        )

    def __len__(self):
        return self.steps

class BertForNer():

    def __init__(self):
        pass

    def load_data(self,data_path):
        """
        # 加载并处理数据
        :param data_path:
        :return: [[['讯', 'O'], ['1月2日', 'TIME'], ['消息，', 'O'], ['滴滴', 'ORG'], ['“金融服务”频道在', 'O']]]
        """
        with open(data_path,'r',encoding='utf-8') as f:
            df = f.read().split('\n\n')
        _texts = []
        for _d in df:
            if _d:
                d, last_flag = [], ''
                for c in _d.split('\n')[:args.MAX_LEN]:
                    char, this_flag = c.split(' ')
                    if this_flag == 'O' and last_flag == 'O':
                        d[-1][0] += char
                    elif this_flag == 'O' and last_flag !='O': # last_flag !='O'也就意味着要么是整个句子的开头，要么是实体之后
                        d.append([char,this_flag])
                    elif this_flag.startswith('B'):
                        d.append([char,this_flag.split('_')[1]])
                    elif this_flag.startswith('I'):
                        d[-1][0] += char
                    last_flag = this_flag # 一行过后更新last_flag
                _texts.append(d)
        return _texts

    def build_model(self):
        """
        建模，加载bert预训练模型，并在最后几层进行微调
        :return:
        """
        bert_model = build_transformer_model(config_path=args.BERT_CONFIG,
                                             checkpoint_path=args.BERT_MODEL)
        output = bert_model.get_layer(args.BERT_LAYER).output
        output = Dropout(rate=0.5)(output)
        output = Dense(_labels_num)(output)
        CRF = ConditionalRandomField(lr_multiplier=1)
        p = CRF(output)
        model = Model(bert_model.input,p)
        model.compile(
            loss=CRF.sparse_loss,
            optimizer=Adam(lr=1e-5),
            metrics=[CRF.sparse_accuracy]
        )
        model.summary()
        return model

    @staticmethod
    def keras_callbacks():
        """
        模型的回调函数
        :return:
        """
        earlystop = EarlyStopping(patience=3,verbose=1,monitor='val_loss')
        checkpoint = ModelCheckpoint(filepath=args.MODEL_PATH,save_best_only=True,save_weights_only=False)
        callbacks = [earlystop,checkpoint]
        return callbacks

    def start_to_train(self):
        # 加载数据
        all_train_data = self.load_data(data_path=args.TEST_DATA_PATH)
        all_test_data = self.load_data(data_path=args.TEST_DATA_PATH)

        # 建模
        model = self.build_model()

        # 回调函数
        callbacks = BertForNer.keras_callbacks()

        # KFold数据
        data_idx = KFold(n_splits=3,shuffle=True).split(all_train_data)

        for train_idx,valid_idx in data_idx:
            train_data = [all_train_data[i] for i in train_idx]
            valid_data = [all_train_data[i] for i in valid_idx]

            # 数据迭代生成器
            train_data_generator = DataGenerator(data=train_data)
            valid_data_generator = DataGenerator(data=valid_data)

            # 训练
            model.fit_generator(
                train_data_generator.__iter__(),
                steps_per_epoch=train_data_generator.__len__(),
                verbose=1,
                callbacks=callbacks,
                epochs=args.EPOCHS,
                validation_data=valid_data_generator.__iter__(),
                validation_steps=valid_data_generator.__len__()
            )

if __name__ == '__main__':

    BertForNer().start_to_train()