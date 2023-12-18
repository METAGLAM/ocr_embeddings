import json
from simalign import SentenceAligner
from Levenshtein import distance as lev

myaligner = SentenceAligner(model="bert", token_type="bpe",
                            matching_methods="mai")


class Line:
    def __init__(self, line=None, load=None):
        if line:
            self.gt = self.remove_weird_chars(line['manually_corrected_line'])
            self.raw = self.remove_weird_chars(line['raw_line'])
            self.pro = self.remove_weird_chars(line['processed_line'])

            self.lev_raw = self.compute_dev(self.gt, self.raw)
            self.lev_pro = self.compute_dev(self.gt, self.pro)

            self.alignments_gt_raw = self.merge_alignments(
                myaligner.get_word_aligns(self.gt, self.raw)['mwmf'])
            self.alignments_gt_pro = self.merge_alignments(
                myaligner.get_word_aligns(self.gt, self.pro)['mwmf'])

            self.raw_misspelled_words = []
            self.pro_misspelled_words = []

            self.joint_alignment = []
            for i, gt in enumerate(self.gt.split()):
                self.joint_alignment.append({
                    'gt': gt,
                    'it_gt': i,
                    'raw': None,
                    'it_raw': None,
                    'miss_raw': None,
                    'pro': None,
                    'it_pro': None,
                    'miss_pro': None
                })
        if load:
            self.gt = self.remove_weird_chars(load['gt'])
            self.raw = self.remove_weird_chars(load['raw'])
            self.pro = self.remove_weird_chars(load['pro'])
            self.lev_raw = self.compute_dev(self.gt, self.raw)
            self.lev_pro = self.compute_dev(self.gt, self.pro)
            self.alignments_gt_raw = load['alignments_gt_raw']
            self.alignments_gt_pro = load['alignments_gt_pro']
            self.raw_misspelled_words = load['raw_misspelled_words']
            self.pro_misspelled_words = load['pro_misspelled_words']
            # self.joint_alignment = load['joint_alignment']
            self.joint_alignment = []
            for i, gt in enumerate(self.gt.split()):
                self.joint_alignment.append({
                    'gt': gt,
                    'it_gt': i,
                    'raw': None,
                    'it_raw': None,
                    'miss_raw': None,
                    'pro': None,
                    'it_pro': None,
                    'miss_pro': None
                })

    @staticmethod
    def remove_weird_chars(line):
        return " ".join(
            line.replace(':', '').replace('"', '').replace(';', '')
            .replace(')', '').replace('(', '').replace('..', '')
            .replace('¿', '').replace('«', '').replace("—", "-")
            .replace("-", "-").replace('»', '').split())

    @staticmethod
    def compute_dev(gt_line, other_line):
        return lev(gt_line.lower(), other_line.lower())

    @staticmethod
    def merge_alignments(alignments_gt):
        # when there are two words belonging to the same GT, we will join them
        # with a blank space in between and treat them as if they were one word
        i = 0
        merged_alignments_gt = []
        while i < len(alignments_gt):
            i2 = i + 1
            while (i2 < len(alignments_gt)) and (
                    alignments_gt[i][0] == alignments_gt[i2][0]):
                i2 += 1

            alignment_list = []
            for j in range(i, i2):
                alignment_list.append(alignments_gt[j][1])

            merged_alignments_gt.append([alignments_gt[i][0], alignment_list])
            i += (i2 - i)

        # when one word raw/pro aligns with a gt, we will not allow it to  align
        # with any other word
        new_merged_alignments_gt = []
        i = 0
        while (new_merged_alignments_gt != merged_alignments_gt):
            if i == 0:
                new_merged_alignments_gt = copy.deepcopy(merged_alignments_gt)
            elif i > 0:
                merged_alignments_gt = copy.deepcopy(new_merged_alignments_gt)

            selected_w = set()

            for _, al in merged_alignments_gt:
                # prioritize words with a unique alignment with a GT
                if len(al) == 1:
                    selected_w.add(al[0])

            for j, (_, al) in enumerate(merged_alignments_gt):
                if len(al) > 1:
                    # drop words with a unique alignment
                    alignment_list = (
                        sorted(list(set(al).difference(selected_w))))
                    new_merged_alignments_gt[j][1] = alignment_list
            i += 1

        return merged_alignments_gt

    def join_alignments(self):
        i_gt = 0
        i_raw = 0
        i_pro = 0
        while i_gt < len(self.joint_alignment):
            if (i_raw < len(self.alignments_gt_raw)) and (
                    self.alignments_gt_raw[i_raw][0] == i_gt):
                self.joint_alignment[i_gt]['raw'] = " ".join(
                    [self.raw.split()[i] for i in
                     self.alignments_gt_raw[i_raw][1]])
                self.joint_alignment[i_gt]['it_raw'] = \
                    self.alignments_gt_raw[i_raw][1]
                self.joint_alignment[i_gt]['miss_raw'] = \
                    0 if self.joint_alignment[i_gt]['gt'].lower() == \
                         self.joint_alignment[i_gt]['raw'].lower() else 1
                i_raw += 1
            if (i_pro < len(self.alignments_gt_pro)) and (
                    self.alignments_gt_pro[i_pro][0] == i_gt):
                self.joint_alignment[i_gt]['pro'] = " ".join(
                    [self.pro.split()[i] for i in
                     self.alignments_gt_pro[i_pro][1]])
                self.joint_alignment[i_gt]['it_pro'] = \
                    self.alignments_gt_pro[i_pro][1]
                self.joint_alignment[i_gt]['miss_pro'] = \
                    0 if self.joint_alignment[i_gt]['gt'].lower() == \
                         self.joint_alignment[i_gt]['pro'].lower() else 1
                i_pro += 1
            i_gt += 1


def write_lines(aligned_lines, filename):
    json_list = []
    for aligned_line in aligned_lines:
        json_line = {
            'gt': aligned_line.gt,
            'raw': aligned_line.raw,
            'pro': aligned_line.pro,
            'lev_raw': aligned_line.lev_raw,
            'lev_pro': aligned_line.lev_pro,
            'alignments_gt_raw': aligned_line.alignments_gt_raw,
            'alignments_gt_pro': aligned_line.alignments_gt_pro,
            'raw_misspelled_words': aligned_line.raw_misspelled_words,
            'pro_misspelled_words': aligned_line.pro_misspelled_words,
            'joint_alignment': aligned_line.joint_alignment
        }
        json_list.append(json_line)

    json.dump(json_list, open(filename, 'w'))


def read_lines(filename):
    return json.load(open(filename))


def check_line(aligned_lines, line_num):
    aligned_line = aligned_lines[line_num]
    print(f"gt:  {aligned_line.gt}")
    print(f"raw: {aligned_line.raw}")
    print(f"pro: {aligned_line.pro}")
    print(format('gt', '15'), format('raw', '17'), format('pro', '15'))
    for word in aligned_line.joint_alignment:
        if word:
            print(format(word['gt'], '15'), format(word['miss_raw'] or '', '1'),
                  format(word['raw'] or '', '15'),
                  format(word['miss_pro'] or '', '1'),
                  format(word['pro'] or '', '15'))
            print(format('', '15'), format(aligned_line.lev_raw, '<17'),
                  format(aligned_line.lev_pro, '<16'))


def extract_counts(aligned_lines):
    raw_correct = 0
    pro_correct = 0
    lev_raw = 0
    lev_pro = 0
    len_char_gt = 0
    len_words_gt = 0
    for line in aligned_lines:
        lev_raw += line.lev_raw
        lev_pro += line.lev_pro
        len_char_gt += len(line.gt)
        len_words_gt += len(line.gt.split())
        for word in line.joint_alignment:
            if word['miss_raw'] == 0:
                raw_correct += 1
            if word['miss_pro'] == 0:
                pro_correct += 1

    return raw_correct, pro_correct, lev_raw, lev_pro, len_char_gt, len_words_gt
