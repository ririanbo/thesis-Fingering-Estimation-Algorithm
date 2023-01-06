
# ここをDPを使った関数に改良する
# 一番低いポジションの運指を選択
# score : # {key : [sound_num, height_g, height_d, height_a, height_e, position, fin_g, fin_d, fin_a, fin_e]}
def select_low_pos(self, score):
    lowest_score = {}
    for key, sheet in score.items():
        sound_num = 0
        options = [] # [height_g, height_d, height_a, height_e]
        new_sheet = [] # [sound_num, height_g, height_d, height_a, height_e]
        # note : [sound_num, pitch, height_g, height_d, height_a, height_e]
        for itr, opt in enumerate(sheet):
            if sound_num == opt[0] and itr != len(sheet)-1:
                options.append(opt)
                continue
            else:
                if itr == len(sheet)-1:
                    options.append(opt)

                min_pos = 26
                lowest_itr = 0
                for i, _opt in enumerate(options):
                    if _opt[5] < min_pos:
                        min_pos = _opt[5]
                        lowest_itr = i

                new_sheet.append(options[lowest_itr])

            options = []
            options.append(opt)
            sound_num += 1

        lowest_score[key[:-4] + '_low.csv'] = new_sheet

    return lowest_score #{key : [sound_num, height_g, height_d, height_a, height_e, position, fin_g, fin_d, fin_a, fin_e]}

