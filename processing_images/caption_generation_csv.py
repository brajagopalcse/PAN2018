import csv


class CaptionGenerationCSV:
    def __init__(self):
        self.image_caption_dict = self.read_csv('./resources/training_image_caption.csv')
        self.image_caption_dict.update(self.read_csv('./resources/test_image_caption.csv'))
        print('Total {} profiles image caption loaded'.format(len(self.image_caption_dict)))

    def read_csv(self, file_name):
        temp_dict = {}
        initial_line = True
        with open(file_name, 'r') as csvfile:
            all_data = csv.reader(csvfile, delimiter=',')
            for row in all_data:
                if initial_line:
                    initial_line = False
                    continue
                i_d = row[0]
                text = row[1]
                if i_d in temp_dict:
                    temp = temp_dict[i_d]
                    temp.append(text)
                    temp_dict[i_d] = temp
                else:
                    temp_dict[i_d] = [text]
        return temp_dict

    def get_caption(self, i_d):
        if i_d in self.image_caption_dict:
            return self.image_caption_dict[i_d]
        else:
            return ''
