import os, sys
from lxml import etree as ET
import codecs


class ReadingNWrite:

    def files_in_folder(self, path):
        xml_file_dict = {}
        image_directory_dict = {}
        labels = {}
        gold_file = ''
        path = os.path.abspath(path) + '/'
        # Differenciate between file
        for file_name in os.listdir(path):
            if '.DS_Store' in file_name:  # This is to handle .DS_Store file
                continue
            if file_name == 'text':
                file_path = path + file_name + '/'
                for xml_file in os.listdir(file_path):
                    file_id = xml_file.replace('.xml', '')
                    if file_id in xml_file_dict:
                        print('duplicate file ' + xml_file)
                        sys.exit(1)
                    xml_file_dict[file_id] = file_path + '/' + xml_file
            elif file_name == 'photo':
                file_path = path + file_name + '/'
                for photo_dir in os.listdir(file_path):
                    photo_path = file_path + photo_dir + '/'
                    if photo_dir in image_directory_dict:
                        print('duplicate file ' + photo_path)
                        sys.exit(1)
                    if '.DS_Store' in photo_dir:
                        continue
                    image_directory_dict[photo_dir] = photo_path
            elif file_name == 'truth.txt':
                gold_file = path + file_name
            else:
                print('Some other file in data folder' + file_name)
        if gold_file == '':
            print('There is no gold file. Is it for test?')
            print('Coping all the keys from xml dict to label dict')
            for file_id in xml_file_dict:
                labels[file_id] = ''
        else:
            # reading gender information
            for line in open(gold_file, 'r'):
                line = line.strip()
                if line == '':
                    print('empty line')
                    sys.exit(1)
                a = line.split(':::')
                if not len(a) == 2:
                    print('More than two items in gold annotation. Check line = ' + line)
                    sys.exit(1)
                labels[a[0]] = a[1]
        if not len(xml_file_dict) == len(image_directory_dict):
            print("Don't have equal number of files")
        return labels, xml_file_dict, image_directory_dict

    def read_files(self, folder_name):
        file_list = []
        for root, dirs, files in os.walk(folder_name):
            for _file_ in files:
                file_list.append(os.path.join(root, _file_))
        return file_list

    def read_tweets(self, file_name):
        tweets = []
        tree = ET.parse(file_name)
        root = tree.getroot()
        for cdata in root.iter('document'):
            tweets.append(cdata.text)
            #print(cdata.text)
        return tweets

    def read_any_list(self, file_name):
        lex_list = []
        with codecs.open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                lex_list.append(line.strip().lower())
        return lex_list

    def format_n_write_output(self, test_output_add, text_labels, image_labels, combined_labels):
        if test_output_add == '':
            print('Output address is null')
            sys.exit(1)
        if not os.path.exists(test_output_add):
            os.mkdirs(test_output_add)
        for author_id in combined_labels:
            author = ET.Element('author')
            author.attrib['id'] = author_id
            author.attrib['lang'] = 'en'
            author.attrib['gender_txt'] = text_labels[author_id]
            author.attrib['gender_img'] = image_labels[author_id]
            author.attrib['gender_comb'] = combined_labels[author_id]
            tree = ET.ElementTree(author)
            tree.write(test_output_add + author_id + '.xml', pretty_print=True)
