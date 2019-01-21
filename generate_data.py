# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/21 下午4:21
 @FileName: generate_data.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import numpy as np

vocab_size = 30000


def write_lst_to_file(lst, filename):
    output = '\n'.join(lst)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(output)


def generate_data():
    np.random.seed(1028)
    for i in range(30):
        data = np.random.random_integers(0, vocab_size - 1, size=[1024*5, 512]).tolist()
        data = [' '.join(list(map(str, x))) for x in data]
        write_lst_to_file(data, 'data/{}.txt'.format(i))


if __name__ == '__main__':
    generate_data()
