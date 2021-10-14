from collections import Counter
import pandas as pd
import numpy as np
import argparse

class KRLawData:
    """ 각 조항마다 빈도가 높게 나타나는 키워드를 분류 모델에 이용하기 위해 
        각 조항에 나타나는 키워드의 빈도를 이용하여 각 키워드마다 임베딩 벡터를 만든다.
    """
    def __init__(self, args):
        if args.base_data == "../data/contracts1-16_with_opt10.csv":
            self.key_list = ['옵션조항1', '옵션조항2', '옵션조항3', '옵션조항5', '옵션조항6', '옵션조항7', '옵션조항8', '옵션조항10',
                            '필수조항1', '필수조항2', '필수조항3', '필수조항4', '필수조항5', '필수조항6', '필수조항7',
                            '필수조항8', '필수조항9', '필수조항10', '필수조항11', '필수조항12', '필수조항13']
        
        # 옵션조항10 없는 경우
        elif args.base_data == "../data/contracts1-16_without_opt10.csv":
            self.key_list = ['옵션조항1', '옵션조항2', '옵션조항3', '옵션조항5', '옵션조항6', '옵션조항7', '옵션조항8',
                            '필수조항1', '필수조항2', '필수조항3', '필수조항4', '필수조항5', '필수조항6', '필수조항7',
                            '필수조항8', '필수조항9', '필수조항10', '필수조항11', '필수조항12', '필수조항13']

        self.label2id = {k:v for v, k in enumerate(self.key_list)}

        # self.df = pd.read_csv("../data/contracts1-15.csv")
        self.df = pd.read_csv(args.base_data)

    def count_dict_by_label(self):
        """ 각 조항별 데이터 개수 확인.
        """
        label_dict = dict()
        for i in range(len(self.df)):
            label = str(self.df.loc[i]['조항구분'])
            if label != 'nan':
                if label not in label_dict:
                    label_dict[label] = 1
                else:
                    label_dict[label] += 1
        ordered_label_dict = {k:v for k, v in sorted(label_dict.items(), key=lambda x: self.label2id[x[0]])}

        return ordered_label_dict

    def use_keyword(self, keyword):
        """ 주어진 keyword가 각 조항마다 나타나는 빈도 수를 계산.
        """
        counter = Counter()
        for i in range(len(self.df)):
            if keyword in str(self.df.iloc[i]['원본']):
                counter.update({str(self.df.iloc[i]['조항구분']): 1})
            
        return counter.most_common()

    def keyword_count_for_each_label(self, keyword):
        """ 주어진 keyword의 각 조항에 대한 분포를 확률로 나타내어 dictionary 형태로 return.

        Example:
            {'필수조항1': 0.8024971623155505, '필수조항2': 0.10457516339869281, 
            '필수조항3': 0.0022026431718061676, '필수조항4': 0.0, '필수조항5': 0.0, '필수조항6': 0.0, 
            '필수조항7': 0.006756756756756757, '필수조항8': 0.016377649325626204, '필수조항9': 0.0, 
            '필수조항10': 0.0, '필수조항10.1': 0.0, '필수조항11': 0.0001808972503617945, '필수조항12': 0.0, '필수조항13': 0.0, '옵션조항1': 0.0, 
            '옵션조항2': 0.0, '옵션조항3': 0.0, '옵션조항4': 0.0, '옵션조항5': 0.0, '옵션조항6': 0.0, 
            '옵션조항7': 0.0, '옵션조항8': 0.0, '옵션조항9': 0.0}
        """
        cnt_dict = {k:v for k, v in zip(self.key_list, [0 for _ in range(len(self.key_list))])}

        ordered_label_dict = self.count_dict_by_label()
        cnt_list = self.use_keyword(keyword)

        for label, cnt in cnt_list:
            if label in cnt_dict:
                cnt_dict[label] += cnt
        
        result_dict = {k:v for k, v in zip(self.key_list, [0 for _ in range(len(self.key_list))])}

        for i, (v1, v2) in enumerate(zip(cnt_dict.values(), ordered_label_dict.values())):
            result_dict[self.key_list[i]] = v1/v2
        
        return result_dict

    def get_keyword_vector(self, keyword):
        """ 주어진 keyword의 각 조항에 대한 분포를 통해 얻은 벡터를 return.
        """
        return np.array(list(self.keyword_count_for_each_label(keyword).values()))

    @staticmethod
    def get_keyword_vector_for_each_text(text, keyword2vec_df):
        """ 주어진 text에 존재하는 keyword2vec_df의 키워드를 찾아 대응되는 벡터를 평균하여 return.
        """
        keywords_in_text = []
        keyword_vector = [0 for _ in range(len(keyword2vec_df.columns))]
        
        # text에 존재하는 키워드 리스트를 저장
        for kw in keyword2vec_df.index:
            if kw in text:
                keywords_in_text.append(kw)

        # keyword vector 계산
        # keyword가 하나 이상 존재하는 경우
        if keywords_in_text:
            for j, kw in enumerate(keywords_in_text):
                if j == 0:
                    tmp_array = np.array(keyword2vec_df.loc[kw])
                    continue
                tmp_array = np.add(tmp_array, np.array(keyword2vec_df.loc[kw]))
            keyword_vector = tmp_array / len(keywords_in_text)
        
        return list(keyword_vector)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Keyword Vectors")

    parser.add_argument("--base_data", dest="base_data", type=str,
                        default="../data/contracts1-16_without_opt10.csv", help="base data for keyword vectors")

    parser.add_argument("--keywords_list", dest="keywords_list", type=str,
                        default="../data/all_keywords.txt", help="keywords list txt file path")
    
    parser.add_argument("--keyword2vec", dest="keyword2vec", type=str,
                        default="../data/keyword2vec_without_opt10.csv", help="keyword embedding csv file path")

    args = parser.parse_args()

    with open(args.keywords_list) as f:
        lines = f.readlines()
    
    all_keywords = [l.strip() for l in lines]
        
    krlawdata = KRLawData(args)
    
    print(f"총 키워드의 개수는 {len(all_keywords)}개입니다!")
    print("Start processing..")

    # 각 키워드의 벡터를 csv 파일로 저장
    kw2vec = dict()
    for i, kw in enumerate(all_keywords):
        kw2vec[kw] = krlawdata.get_keyword_vector(kw)
        if i % 30 == 29:
            print(f"{i+1}개 키워드 완료!")
    print(f"********** 총 {len(all_keywords)}개 키워드 벡터 저장 완료 **********")
    
    kw2vec_df = pd.DataFrame.from_dict(kw2vec, orient='index')
    kw2vec_df.to_csv(args.keyword2vec, index_label='키워드')