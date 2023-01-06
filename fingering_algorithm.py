import numpy as np
import os
import math
import csv


class fingering_algorithm(object):
    def __init__(self):
        self.pitch_for_invalid_note = 101
        self.lowest_pitch = 55
        self.n_p_classes = 46 + 1  # pitch range =  55 to 100, pitch_for_invalid_note = 101
        self.n_b_classes = 7  # {'', '1th', '2th', '4th',  '8th',  '16th', '32th'}
        self.n_str_classes = 5  # {'', G', 'D', 'A', 'E'}
        self.n_pos_classes = 13  # {'', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'}
        self.n_fin_classes = 5  # {'' & '0', '1', '2', '3', '4'}
        self.string_height_classes = 26  # 弦上で押さえられる場所（0～25）
        self.hidden_size = 100
        self.INFIN = 1e12  # infinite number
        self.dataset_dir = "./double_stop_dataset"
        self.result_dir = "./result"
        self.shaped_dir = "./shaped_result"
        self.kaiho = [55, 62, 69, 76] #開放弦のピッチ
        self.source_dir = "./TNUA_violin_fingering_dataset"
        self.dist_dir = "./TNUA_violin_fingering_dataset_formed"
        self.compare_dir = "./compare_fingering"
        self.music_name = ["bach1", "bach2", "beeth1", "beeth2_1", "beeth2_2", "elgar", "flower","mend1", "mend2", "mend3", "mozart1", "mozart2_1", "mozart2_2", "wind", "capriccioso1", "grade_3_b", "grade_4_a", "grade_5_a"]


    def get_unique_list(self, seq):
        seen = []
        return [x for x in seq if x not in seen and not seen.append(x)]


    def load_data(self):
        print("Load input data...")
        files = [x for x in os.listdir(self.dataset_dir) if x.endswith('csv')]
        corpus = {}
        for file in files:
            with open(self.dataset_dir + '/' + file, encoding='utf8') as f:
                corpus[file] = np.genfromtxt(f, delimiter=',', names=True, dtype=[('int'), ('float')], usecols=[1, 2])  # pitch, start, measure

        print("Loading finished")
        return corpus

    # [pitch, start, duration, beat_type, string, position, finger] → [ _, pitch, start]
    # start == sound_num,
    def change_form(self):
        files = [x for x in os.listdir(self.source_dir) if x.endswith('csv')]
        corpus = {}
        for file in files:
            with open(self.source_dir + '/' + file, encoding='utf8') as f:
                corpus[file] = np.genfromtxt(f, delimiter=',', names=True,
                                             dtype=[('int'), ('float'), ('float'), ('int'), ('int'), ('int'), ('int')])

            with open(self.dist_dir + '/' + file, 'w', encoding='utf8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["", "pitch", "start"])
                for note in corpus[file]:
                    writer.writerow(["", note[0], note[1]])


    # return where to put (string, place) to make the pitch
    # corpus : {key : [pitch, start]}
    def where_to_put(self, corpus):
        score = {}

        for key, notes in corpus.items():
            pre_start = notes[0][1]
            sound_num = 0
            sheet = []
            for note in notes:
                pitch = note[0]
                start = note[1]
                if pre_start != start:
                    sound_num += 1
                    pre_start = start

                if pitch >= 55 and pitch <= 79:
                    height_g = pitch - 55
                else:
                    height_g = -1

                if pitch >= 62 and pitch <= 86:
                    height_d = pitch - 62
                else:
                    height_d = -1

                if pitch >= 69 and pitch <= 93:
                    height_a = pitch - 69
                else:
                    height_a = -1

                if pitch >= 76 and pitch <= 100:
                    height_e = pitch - 76
                else:
                    height_e = -1

                #
                # if (height_g < 0 or 24 < height_g):
                #     height_g = -1
                # if (height_d < 0 or 24 < height_d):
                #     height_d = -1
                # if (height_a < 0 or 24 < height_a):
                #     height_a = -1
                # if (height_e < 0 or 24 < height_e):
                #     height_e = -1
                sheet.append([sound_num, pitch, height_g, height_d, height_a, height_e])
            score[key] = sheet
        return score # {key : [sound_num, pitch, height_g, height_d, height_a, height_e]}


    # 演奏可能な運指の選択肢を全通り出力
    # score : {key : [sound_num, pitch, height_g, height_d, height_a, height_e]}
    def narrow_choice(self, score):
        print("start narrow_choice")
        double_stop_score = {}
        for key, sheet in score.items():
            print(key)
            sound_num = 0
            double_stop = [] # [[height_g, height_d, height_a, height_e], ...]
            place_set = [] # [sound_num, height_g, height_d, height_a, height_e]
            # note : [sound_num, pitch, height_g, height_d, height_a, height_e]
            for itr, note in enumerate(sheet):
                if itr % 1000 == 0: print(itr)
                if(sound_num == note[0] and itr != len(sheet)-1):
                    double_stop.append(note[2:])
                    continue
                else:
                    if(itr == len(sheet)-1):
                        double_stop.append(note[2:])
                    # test
                    # if itr <= 10:
                    #     print(double_stop) # 重音ごとに分けてリスト化した物の表示
                    length = len(double_stop)
                    # bubble sort 的に音の順番を入れ替えて押さえ方の選択しをすべて考慮する
                    fact = math.factorial(length)
                    for comb in range(fact):
                        if(comb != 0):
                            # swap
                            tmp = comb % length
                            double_stop[tmp-1], double_stop[tmp] = double_stop[tmp], double_stop[tmp-1]

                        place_low = [sound_num, -1, -1, -1, -1]
                        flag_low = [0, 0, 0, 0]  # すでに選んだ弦を保存

                        place_high = [sound_num, -1, -1, -1, -1]
                        flag_high = [0, 0, 0, 0]  # すでに選んだ弦を保存

                        place_middle = [sound_num, -1, -1, -1, -1]
                        flag_middle = [0, 0, 0, 0]  # すでに選んだ弦を保存

                        place_middle2 = [sound_num, -1, -1, -1, -1]
                        flag_middle2 = [0, 0, 0, 0]  # すでに選んだ弦を保存

                        for j in range(length):
                            # 下の弦から決めていく
                            for k in range(4): # 4回繰り返し
                                # 既にその弦を使っている or その弦で出せない音ならskip
                                if(flag_low[k] == 1 or double_stop[j][k] == -1):
                                    continue
                                # その弦がその音に使えるなら押さえる位置を占有
                                else:
                                    place_low[k + 1] = double_stop[j][k]
                                    flag_low[k] = 1
                                    break
                            # 上の弦から決めていく、一番低いポジションの選択肢が出現
                            for k in range(4): # 4回繰り返し
                                k = 3 - k
                                # 既にその弦を使っている or その弦で出せない音ならskip
                                if(flag_high[k] == 1 or double_stop[j][k] == -1):
                                    continue
                                # その弦がその音に使えるなら押さえる位置を占有
                                else:
                                    place_high[k + 1] = double_stop[j][k]
                                    flag_high[k] = 1
                                    break

                            # 中央の弦から決めていく
                            for k in range(4): # 4回繰り返し
                                k = (k+1) % 4
                                # 既にその弦を使っている or その弦で出せない音ならskip
                                if(flag_middle[k] == 1 or double_stop[j][k] == -1):
                                    continue
                                # その弦がその音に使えるなら押さえる位置を占有
                                else:
                                    place_middle[k + 1] = double_stop[j][k]
                                    flag_middle[k] = 1
                                    break

                            # 中央の弦から決めていく_2
                            for k in range(4): # 4回繰り返し
                                k = (k+2) % 4
                                # 既にその弦を使っている or その弦で出せない音ならskip
                                if(flag_middle2[k] == 1 or double_stop[j][k] == -1):
                                    continue
                                # その弦がその音に使えるなら押さえる位置を占有
                                else:
                                    place_middle2[k + 1] = double_stop[j][k]
                                    flag_middle2[k] = 1
                                    break

                        # 重音の全ての音について指を割り当てたら
                        if(flag_low.count(1) == length):
                            place_set.append(place_low)
                        if(flag_high.count(1) == length):
                            place_set.append(place_high)
                        if(flag_middle.count(1) == length):
                            place_set.append(place_middle)
                        if (flag_middle2.count(1) == length):
                            place_set.append(place_middle2)

                double_stop = []
                double_stop.append(note[2:])
                sound_num += 1
            # 重複の排除
            place_set = self.get_unique_list(place_set)

            # 弦の飛びを排除
            for i, place in enumerate(place_set):
                flag = 0
                for pos in place[1:]:
                    if flag == 0 and pos != -1:
                        flag += 1
                    if flag == 1 and pos == -1:
                        flag += 1
                    if flag == 2 and pos != -1:
                        place_set[i] = []

            # 手のサイズ（6 pitch）より大きい押さえ方を排除
            for i, place in enumerate(place_set):
                max = -1
                min = 26
                for pos in place[1:]:
                    if pos <= 0:
                        continue
                    if min > pos:
                        min = pos
                    if max < pos:
                        max = pos

                if max - min > 6:
                    place_set[i] = []

            place_set = [i for i in place_set if i != []]

            double_stop_score[key] = place_set
            # # test
            # for i, p in enumerate(place_set):
            #     print(p)
            #     if i == 100:
            #         break

        print("narrow_choice finished")

        return double_stop_score #{key : [sound_num, g, d, a, e]}

    # place : [height_g, height_d, height_a, height_e, position]
    def predict_finger(self, place):
        g_str = [0, 2, 4, 5, 7, 9, 10, 12, 14, 16, 17, 19, 21, 22, 24, 26]
        d_str = [0, 2, 3, 5, 7, 9, 10, 12, 14, 15, 17, 19, 21, 22, 24, 26]
        a_str = [0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 26]
        e_str = [0, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 19, 20, 22, 24, 25]
        position = place[4]

        fingering = [[] for i in range(4)]  # [fin_g, fin_d, fin_a, fin_e]
        base_fin1 = [g_str[position], d_str[position], a_str[position], e_str[position]]

        # 各指の守備範囲
        base_roll = [[-1, 0],
                     [0, 1, 2],
                     [1, 2, 3, 4],
                     [2, 3, 4, 5]]
        opt_num = 0  # 運指の選択肢数
        # g線から順に決定していく
        for i, height in enumerate(place[:4]):
            # 開放弦or弾かない弦だったら特に何もしない
            if height <= 0:
                fingering[i].append(height)
                continue

            for j, option in enumerate(base_roll):
                option = map(lambda val: val + base_fin1[i], option)
                for k, opt in enumerate(option):
                    if opt == height:
                        fingering[i].append(j + 1)
        N = 1
        for fin in fingering:
            N *= len(fin)
            # 押さえられる指が無かったら
            if N == 0:
                return []
        fingering_all = [[-1] * 4 for i in range(N)]
        # 全通りの運指を出力
        i = 0
        for fin_g in fingering[0]:
            for fin_d in fingering[1]:
                for fin_a in fingering[2]:
                    for fin_e in fingering[3]:
                        fingering_all[i] = [fin_g, fin_d, fin_a, fin_e]
                        i += 1

        return fingering_all # [[fin_g, fin_d, fin_a, fin_e], ...]

        # base_position = [g_str[position:position + 5], d_str[position:position + 5], a_str[position:position + 5],
            #                  e_str[position:position + 5]]

            # # このままだと指の選択が現実的でない
            # for j, pos in enumerate(base_position[i]):
            #     # 4の指で押さえきれない場合
            #     if j == 4:
            #         return []
            #     if height <= pos:
            #         fingering.append(j + 1)
            #         break




    # 選択肢の幅だしをする(考えられる弦位置、ポジションを全て出力)
    # score : {key : [sound_num, g, d, a, e]}
    def possible_options(self, score):
        # 弦のポジションごとの最低ピッチ
        g_str = [0, 2, 4, 5, 7, 9, 10, 12, 14, 16, 17, 19, 21, 22, 24, 26]
        d_str = [0, 2, 3, 5, 7, 9, 10, 12, 14, 15, 17, 19, 21, 22, 24, 26]
        a_str = [0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 26]
        e_str = [0, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 19, 20, 22, 24, 25]
        base_positions = [g_str, d_str, a_str, e_str]
        fingering_options_score = {} # [sound_num, height_g, height_d, height_a, height_e, position, fin_g, fin_d, fin_a, fin_e]

        for key, sheet in score.items():
            new_sheet = []
            for test, opt in enumerate(sheet):
                min = 26
                max = -1
                min_str = 0
                for i, pos in enumerate(opt[1:]):
                    if pos <= 0: # ここじゃない？
                        continue
                    if pos > max:
                        max = pos
                    if pos < min:
                        min = pos
                        min_str = i
                # 弦の最高位置と最低位置のピッチ幅
                width = max - min

                if width >= 4:
                    for base, height in enumerate(base_positions[min_str]):
                        if base > 12:
                            break
                        if min <= height:
                            # 最低音を1で弾く
                            fin = self.predict_finger(opt[1:] + [base])
                            if fin != []:
                                for f in fin:
                                    new_sheet.append(opt + [base] + f)
                            break

                if width == 3 or width == 4:
                    for base, height in enumerate(base_positions[min_str]):
                        if base > 12:
                            break
                        if min <= height:
                            # 最低音を1で弾く
                            fin = self.predict_finger(opt[1:] + [base])
                            if fin != []:
                                for f in fin:
                                    new_sheet.append(opt + [base] + f)
                            if base > 1:
                                # 最低音を2で弾く
                                fin = self.predict_finger(opt[1:] + [base-1])
                                if fin != []:
                                    for f in fin:
                                        new_sheet.append(opt + [base-1] + f)
                            break

                if width == 1 or width == 2:
                    for base, height in enumerate(base_positions[min_str]):
                        if base > 12:
                            break
                        if min <= height:
                            # 最低音を1で弾く
                            fin = self.predict_finger(opt[1:] + [base])
                            if fin != []:
                                for f in fin:
                                    new_sheet.append(opt + [base] + f)
                            if base > 1:
                                # 最低音を2で弾く
                                fin = self.predict_finger(opt[1:] + [base-1])
                                if fin != []:
                                    for f in fin:
                                        new_sheet.append(opt + [base-1] + f)
                            if base > 2:
                                # 最低音を3で弾く
                                fin = self.predict_finger(opt[1:] + [base-2])
                                if fin != []:
                                    for f in fin:
                                        new_sheet.append(opt + [base-2] + f)
                            break


                # 単音の場合
                if width == 0:
                    for base, height in enumerate(base_positions[min_str]):
                        if base > 12:
                            break
                        if min <= height:
                            # 最低音を1で弾く
                            fin = self.predict_finger(opt[1:] + [base])
                            if fin != []:
                                for f in fin:
                                    new_sheet.append(opt + [base] + f)
                            if base > 1:
                                # 最低音を2で弾く
                                fin = self.predict_finger(opt[1:] + [base-1])
                                if fin != []:
                                    for f in fin:
                                        new_sheet.append(opt + [base-1] + f)
                            if base > 2:
                                # 最低音を3で弾く
                                fin = self.predict_finger(opt[1:] + [base-2])
                                if fin != []:
                                    for f in fin:
                                        new_sheet.append(opt + [base-2] + f)
                            if base > 3:
                                # 最低音を4で弾く
                                fin = self.predict_finger(opt[1:] + [base-3])
                                if fin != []:
                                    for f in fin:
                                        new_sheet.append(opt + [base-3] + f)
                            break

            fingering_options_score[key] = new_sheet
        return fingering_options_score # [sound_num, height_g, height_d, height_a, height_e, position, fin_g, fin_d, fin_a, fin_e]

    # evaluateの手前で結果を保存しておく
    def export_possible_options(self, po):
        for key, sheet in po.items():
            with open("./possible_options" + '/' + key[:-4] + "_po.csv", 'w', encoding='utf8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["sound_num", "height_g", "height_d", "height_a", "height_e", "position", "fin_g", "fin_d", "fin_a", "fin_e"])
                for note in sheet:
                    writer.writerow(note)
        print("export completed!")

    def import_possible_options(self):
        files = [x for x in os.listdir("./possible_options") if x.endswith('csv')]
        po = {}

        for file in files:
            new_name = file[:-7] + ".csv"
            po[new_name] = []
            with open("./possible_options" + '/' + file, encoding='utf8', newline='') as f:
                csvreader = csv.reader(f)
                header = next(csvreader)
                for i, row in enumerate(csvreader):
                    row = [int(x) for x in row]
                    po[new_name].append(row)
        print("import completed!")
        return po


    # ポジション、重みの強さ
    def evaluate_position(self, position):
        # base_point = ((self.n_pos_classes - position) * (self.n_pos_classes - position)) / 2
        # base_point = [-1, 72, 60.5, 50, 40.5, 32, 24.5, 18, 12.5, 8, 4.5, 2, 0.5]
        # test
        # base_point = [-1, 72, 60, 61, 40, 41, 32, 33, 18, 19, 12, 13, 0]
        base_point = [-1, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
        h = base_point[position]
        # 1st ポジション優先
        if position == 1:
            h += 10
        # # 3rdポジションを優先
        if position == 3:
            h += 10
        # 奇数ポジションを優先
        if position != 1 and position % 2 == 1:
            h += 5

        # # 初心者
        # if position == 1:
        #     h += 10

        return h


    def evaluate_left_hand_movement(self, pre_position, position):
        distance = abs(pre_position - position)
        h = 0
        # if distance == 0:
        #     return 0
        # else:
        #     h -= ((distance + 1) / 2) * 5

        h -= distance * 3


        return h # [0, -5, -10, -15, -20, -25, ...]


    # 移弦の幅について評価
    def evaluate_right_hand_movement(self, pre_finger, finger):
        h = 0
        # どの弦を弾いているか
        pre_str = 0 # 0 ~ 3
        str = 0 # 0 ~ 3
        str_num = 0
        for i, pre_fin in enumerate(pre_finger):
            if pre_fin != -1:
                pre_str += i
                str_num += 1
        pre_str /= str_num

        str_num = 0
        for i, fin in enumerate(finger):
            if fin != -1:
                str += i
                str_num += 1
        str /= str_num

        distance = abs(str - pre_str)

        h = (-5) * distance

        return h
    # 指だけではなく、押さえる位置と指の関係を併せて考えた方がよさそう
    # height : [height_g, height_d, height_a, height_e], finger : [fin_g, fin_d, fin_a, fin_e]
    def evaluate_finger(self, height, finger, position):
        h = 0

        g_str = [0, 2, 4, 5, 7, 9, 10, 12, 14, 16, 17, 19, 21, 22, 24, 26]
        d_str = [0, 2, 3, 5, 7, 9, 10, 12, 14, 15, 17, 19, 21, 22, 24, 26]
        a_str = [0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 26]
        e_str = [0, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 19, 20, 22, 24, 25]
        # kaiho = [55, 62, 69, 76]
        # flageolet = [67, 74, 81, 88]
        base_positions = [g_str, d_str, a_str, e_str]

        # ノーマルな指の置き方に加点する
        # 各指のノーマル守備範囲(どの弦かに依存しない)
        base_roll = [[-1, 0], # 1の指
                     [1, 2],  # 2の指
                     [3, 4],  # 3の指
                     [4, 5]]  # 4の指
        low_str = low = 30
        for i in range(4):
            if low > height[i]:
                low = height[i]
                low_str = i

        num_of_note = 0
        # 守備範囲をポジションに対応付け
        for i in range(4):
            for j in range(2):
                base_roll[i][j] += base_positions[low_str][position]

            # 単音かどうかの判定
            if finger[i] != -1:
                num_of_note += 1


        for i, fin in enumerate(finger):
            # 開放弦を使っていたらGood
            if fin == 0:
                # 初級者
                h += 10
                # # 中級者以上
                # h += 6
            else:
                # 同じ指を3回以上使うたびに減点
                for fin2 in finger[i+2:]:
                    if fin2 == fin:
                        h -= 10

                if i <= 2 and fin == finger[i+1] and height[i] != height[i+1]:
                    h -= 10

                # ノーマル守備範囲なら加点
                if base_roll[fin-1][0] == height[i] or base_roll[fin-1][1] == height[i]:
                    h += 5



                # 単音かつ自然フラジオレットなら加点
                if num_of_note == 1 and height[i] == 12:
                    h += 7



        return h


    # [sound_num, height_g, height_d, height_a, height_e, position, fin_g, fin_d, fin_a, fin_e]
    def evaluate(self, pre_double_stop, double_stop):
        h = 0
        h += self.evaluate_position(double_stop[5])
        h += self.evaluate_finger(double_stop[1:5], double_stop[6:10], double_stop[5])
        if pre_double_stop != []:
            h += self.evaluate_left_hand_movement(pre_double_stop[5], double_stop[5])
            h += self.evaluate_right_hand_movement(pre_double_stop[6:10], double_stop[6:10])

        return h


    # score : { key : [sound_num, height_g, height_d, height_a, height_e, position, fin_g, fin_d, fin_a, fin_e]}
    def dp(self, score):
        print("finding best route...")
        # dp : 2次元, 重音数 * 運指候補( <= 16)
        # O(重音数 * 運指候補数^2)
        dp_score = {} # {key : dp}
        dp_route = {} # {key : route}

        for key, sheet in score.items():
            print(key)
            sound_num = 0
            pre_double_stop = []
            double_stop = []
            dp = [[-1] * 150 for _ in range(sheet[-1][0]+1)] # [音番号] * [選択肢], 得点
            route = [[-1] * 150 for _ in range(sheet[-1][0] + 1)] # 1つ前で何番を選んだか
            for itr, note in enumerate(sheet):
                # 前の重音の一部 かつ 最後の音でない場合
                if (sound_num == note[0] and itr != len(sheet) - 1):
                    double_stop.append(note)
                    continue
                else:
                    if (sound_num == note[0] and itr == len(sheet) - 1):
                        double_stop.append(note)

                    length = len(double_stop)
                    pre_length = len(pre_double_stop)
                    min = 1000
                    for i in range(length):
                        if sound_num == 0:
                            dp[sound_num][i] = self.evaluate([], double_stop[i]) + 1000
                            continue
                        max = 0
                        max_j = 0
                        for j in range(pre_length):
                            ev = dp[sound_num - 1][j] + self.evaluate(pre_double_stop[j], double_stop[i])
                            if max < ev:
                                max = ev
                                max_j = j

                        dp[sound_num][i] = max
                        route[sound_num][i] = max_j

                        if min > max:
                            min = max

                    # 報酬合計値が大きくなりすぎないようにする
                    for i in range(length):
                        dp[sound_num][i] -= min


                    # 最後の重音の選択肢が1つだった場合の例外処理
                    if (sound_num != note[0] and itr == len(sheet) - 1):
                        pre_double_stop = double_stop
                        ds = note
                        sound_num += 1
                        max = 0
                        max_j = 0
                        for j in range(pre_length):
                            ev = dp[sound_num - 1][j] + self.evaluate(pre_double_stop[j], ds)
                            if max < ev:
                                max = ev
                                max_j = j
                        dp[sound_num][0] = max
                        route[sound_num][0] = max_j
                    sound_num += 1
                    pre_double_stop = double_stop

                    double_stop = []
                    double_stop.append(note)


            dp_score[key] = dp
            dp_route[key] = route
        return dp_score, dp_route


    def find_route(self, dp_score, dp_route):
        last_option = {} # {key : int} # 最後の重音をどの選択肢で弾くのが最大
        route = {}
        for key, sheet in dp_score.items():
            max = -100
            for i, point in enumerate(sheet[-1]):
                if max < point:
                    max = point
                    last_option[key] = i

            # print(last_option[key])

        for key, sheet in dp_route.items():
            num = last_option[key]
            route[key] = [num]
            for options in reversed(sheet):
                route[key].append(options[num])
                num = options[num]
            route[key].reverse()

        print("finished finding route!")
        return route # route[i] = 1個前の選択肢のうちどれを選んだか


    # po_score : { key : [sound_num, height_g, height_d, height_a, height_e, position, fin_g, fin_d, fin_a, fin_e]}
    # route : {key : [1つ前の選択肢]}
    def get_final_score(self, po_score, route):
        final_score = {}

        for key, sheet in po_score.items():
            new_key = key[:-4] + "_optimized.csv"
            sound_num = 0
            option_num = 0
            final_score[new_key] = []
            for itr, note in enumerate(sheet):
                # 前の重音の続き
                if sound_num == note[0]:
                    if route[key][sound_num+1] == option_num:
                        final_score[new_key].append(note)
                    option_num += 1
                    continue

                #新しい重音
                else:
                    sound_num += 1
                    option_num = 0
                    if route[key][sound_num+1] == option_num:
                        final_score[new_key].append(note)
                    option_num += 1

        return final_score


    def print_score(self, score):
        for key, sheet in score.items():
            print(key)
            print("[sound_num, pitch, height_g, height_d, height_a, height_e]")
            print(*sheet, sep='\n')


    def print_ds_score(self, score):
        for key, sheet in score.items():
            print(key)
            print("[sound_num, height_g, height_d, height_a, height_e]")
            print(*sheet, sep='\n')


    def print_p_and_f(self, score):
        for key, sheet in score.items():
            print(key)
            print("[sound_num, height_g, height_d, height_a, height_e, position, fin_g, fin_d, fin_a, fin_e]")
            for i, note in enumerate(sheet):
                print(note)
                if i == 200:
                    break


    def print_note(self, score):
        for key, sheet in score.items():
            print(key)
            print("[sound_num, pitch, string, position, finger]")
            print(*sheet, sep='\n')


    def print_dp_score(self, score):
        for key, dp in score.items():
            print("dp : " + key)
            for i, note in enumerate(dp):
                print([i] + note)
                if i == 10: break


    def print_dp_route(self, route):
        for key, r in route.items():
            print("route : " + key)
            for i, v in enumerate(r):
                print([i] + v)
                if i == 10: break


    # {key : [sound_num, height_g, height_d, height_a, height_e, position, fin_g, fin_d, fin_a, fin_e]}
    def divide_into_note(self_, score):
        note_score = {}
        for key, sheet in score.items():
            new_sheet = []
            for note in sheet:
                string = ['g', 'd', 'a', 'e']
                for i, pos in enumerate(note[1:5]):
                    if pos != -1:

                        pitch = pos + 55 + 7 * i
                        new_sheet.append([note[0]] + [pitch] + [string[i]] + [note[5]] + [note[i+6]])
                    # else:
                    #     pitch[i] = -1

            note_score[key] = new_sheet

        return note_score # {key : [sound_num, pitch, string, position, finger]}


    # score : {key : [sound_num, pitch, string, position, finger]}
    def export_result(self, score):
        for key, sheet in score.items():
            with open(self.result_dir + '/' + key, 'w', encoding='utf8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["sound_num", "pitch", "string", "position", "finger"])
                for note in sheet:
                    writer.writerow(note)


    # score : {key : [sound_num, pitch, string, position, finger]}
    def export_shaped_result(self, score):
        for key, sheet in score.items():
            with open(self.compare_dir + '/result/' + key, 'w', encoding='utf8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["pitch", "start", "duration", "beat_type", "string", "position", "finger"])
                for note in sheet:
                    string = []
                    if note[2] == 'g': string = 1
                    if note[2] == 'd': string = 2
                    if note[2] == 'a': string = 3
                    if note[2] == 'e': string = 4

                    writer.writerow([note[1], note[0], 0.5, 4, string, note[3], note[4]])


    def compare(self):

        target_files = [x for x in os.listdir(self.compare_dir + "/target/") if x.endswith('csv')]
        targets = {}
        for file in target_files:
            targets[file] = []
            with open(self.compare_dir + "/target/" + file, encoding='utf8', newline='') as f:
                csvreader = csv.reader(f)
                header = next(csvreader)
                for i, row in enumerate(csvreader):
                    row[4:7] = [int(x) for x in row[4:7]]
                    targets[file].append(row[4:7])

        result_files = [x for x in os.listdir(self.compare_dir + "/result/") if x.endswith('csv')]
        results = {}
        # 楽曲リスト
        music = [""] * len(result_files)
        m = 0
        for file in result_files:
            for n in self.music_name:
                if n in file:
                    music[m] = n
                    break
            if music[m] == "":
                music[m] = ("NA")
            results[file] = []
            with open(self.compare_dir + "/result/" + file, encoding='utf8', newline='') as f:
                csvreader = csv.reader(f)
                header = next(csvreader)
                for i, row in enumerate(csvreader):
                    row[4:7] = [int(x) for x in row[4:7]]
                    results[file].append(row[4:7])
            m += 1

        m = 0
        s_accuracy = {}
        p_accuracy = {}
        f_accuracy = {}
        accuracy = {}
        print(music)
        for key_r, result in results.items():
            for key_t, target in targets.items():
                if music[m] in key_t:
                    sacc = 0
                    pacc = 0
                    facc = 0
                    acc = 0
                    for i, r in enumerate(result):
                        if i >= len(target):
                            break
                        if r[0] == target[i][0]:
                            sacc += 1
                        if r[1] == target[i][1]:
                            pacc += 1
                        if r[2] == target[i][2]:
                            facc += 1
                        if r[0] == target[i][0] and r[1] == target[i][1] and r[2] == target[i][2]:
                            acc += 1

                    s_accuracy[key_t] = sacc
                    p_accuracy[key_t] = pacc
                    f_accuracy[key_t] = facc
                    accuracy[key_t] = acc

            m += 1


        with open(self.compare_dir + '/comparison/' + "comparison.csv", 'w', encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["target", "String Accuracy", "Position Accuracy", "Fingering Accuracy", "Perfect Accuracy"])
            for key, acc in s_accuracy.items():
                writer.writerow([key, s_accuracy[key], p_accuracy[key], f_accuracy[key], accuracy[key]])
                print(key)
                # F1の値
                print("String Accuracy:", s_accuracy[key])
                print("Position Accuracy:", p_accuracy[key])
                print("Fingering Accuracy:", f_accuracy[key])
                print("Perfect Accuracy:", accuracy[key])


    def do_po(self):
        corpus = self.load_data()

        score = self.where_to_put(corpus)

        double_stop_score = self.narrow_choice(score)

        po = self.possible_options(double_stop_score)

        self.export_possible_options(po)
        # print("export_file")
        # self.print_p_and_f(po)


    def get_result(self):

        # self.print_p_and_f(po)

        po = self.import_possible_options()
        print("import_file")
        # self.print_p_and_f(po)


        dp_score, dp_route = self.dp(po)
        # self.print_dp_score(dp_score)
        # self.print_dp_route(dp_route)
        route = self.find_route(dp_score, dp_route)
        # #test
        # print("route")
        # for key, r in route.items():
        #     print(r)

        final_score = self.get_final_score(po, route)

        notes = self.divide_into_note(final_score)
        # self.print_note(notes)

        self.export_result(notes)
        self.export_shaped_result(notes)


if __name__ == "__main__":
    model = fingering_algorithm()
    # model.do_po()
    model.get_result()
    model.compare()
    # model.change_form()


