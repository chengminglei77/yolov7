import os

directorys = [
    "E:\\datasets\\zh_extra\\anquanmao\\labels",
    # "E:\\datasets\\zh_extra\\chouyan\\labels",
    # "E:\\datasets\\zh_extra\\leifengmao\\labels",
    # "E:\\datasets\\zh_extra\\person\\person1\\labels",
    # "E:\\datasets\\zh_extra\\person\\person2\\labels",
    # "E:\\datasets\\zh_extra\\person\\person3\\labels",
    # "E:\\datasets\\zh_extra\\person\\person4\\labels",
    # "E:\\datasets\\zh_extra\\phone\\labels",
    # "E:\\datasets\\zh_extra\\safety_belt\\labels",
    # "E:\\datasets\\zh_extra\\shuaidao\\labels",
    # "E:\\datasets\\zh_extra\\sleeping\\labels",
    # "E:\\datasets\\zh_extra\\yanwu\\labels",
    # "E:\\datasets\\zh_extra\\jiyouxielou\\labels",
]  # 目录路径
# 2>6    3>8
is_all_change = False
label_raw_value = "14"  # 原始标签
label_value = "8"  # 新标签值
distinct_labels = set()
del_directorys = [
    'images\\project\\project_zhonghan\\01\\labels',
    'images\\project\\project_zhonghan\\02\\labels',
    'images\\project\\project_zhonghan\\03\\labels',
    'images\\project\\project_zhonghan\\04\\labels',
    'images\\project\\project_zhonghan\\cap0219\\labels',
    'images\\project\\project_zhonghan\\caps0226\\labels',

    'images\\behavior\\behavior_call\\labels',
    'images\\behavior\\behavior_call_self\\call\\labels',
    'images\\behavior\\behavior_call_self\\look\\labels',
    'images\\behavior\\behavior_smoking\\labels',
    'images\\behavior\\behavior_tripped\\labels',
    'images\\behavior\\behavior_sleeping\\labels',

    'images\\alarm\\alarm_fire\\labels',
    'images\\alarm\\alarm_fire_self\\01\\labels',
    'images\\alarm\\alarm_fumes\\labels',

    # 误报标签
    'images\\TrainData\\cloud\\labels',
    'images\\TrainData\\lights\\labels',

    'images\\person\\person_self\\person\\labels',
    'images\\person\\person_self\\self\\labels',

    'images\\person\\gong_fu\\labels',  # 工服 反光衣

    'images\\car\\bus\\labels',
    'images\\car\\family_sedan\\labels',
    'images\\car\\fire_engine\\labels',
    'images\\car\\heavy_truck\\labels',
    'images\\car\\jeep\\labels',
    'images\\car\\minibus\\labels',
    'images\\car\\oil_tank_truck\\labels',
    'images\\car\\SUV\\labels',
    'images\\car\\taxi\\labels',
    'images\\car\\truck\\labels',
    'images\\car\\truck_box\\labels',
    'images\\car\\truck_box_simulated\\labels',
    'images\\car\\car_1\\labels',
    'images\\car\\car_2\\labels',
    'images\\car\\car_3\\labels',
    'images\\project\\zh_car\\labels',
    'images\\person\\cap\\labels',  # 安全帽
    'images\\alarm\\alarm_fire01\\labels',
    'images\\alarm\\alarm_fire02\\labels',
    'images\\alarm\\alarm_fire03\\labels',
    'images\\alarm\\alarm_fumes_high_altitude\\labels',
    'images\\alarm\\alarm_fumes_ps_5000\\labels',
    'images\\alarm\\alarm_fumes_ps_6679\\labels',
    'images\\alarm\\alarm_fumes_ps_10000\\labels',

    'images\\person\\person\\labels',
    'images\\person\\coco80_person_14115\\labels',

    'images\\project\\project_0001\\labels'
]

label_remove_value = ['0', '1', '2', '4', '6', '7', '9', '10', '11', '14']
keep_labels = ['0', '1', '2', '4', '6', '7', '9', '10', '11', '14']


def replace_label():
    for directory in directorys:
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                temp_filepath = os.path.join(directory, f"{os.path.splitext(filename)[0]}_temp.txt")

                with open(filepath, "r") as file:
                    lines = file.readlines()

                with open(temp_filepath, "w") as temp_file:
                    for line in lines:
                        split_string = line.split(" ")
                        # 判断是否与原始标签一致
                        if is_all_change == True or split_string[0] == label_raw_value:
                            split_string[0] = label_value
                        modified_line = " ".join(split_string)
                        temp_file.write(modified_line)

                os.remove(filepath)
                os.rename(temp_filepath, filepath)


def remove_label():
    for directory in directorys:
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                temp_filepath = os.path.join(directory, f"{os.path.splitext(filename)[0]}_temp.txt")

                with open(filepath, "r") as file:
                    lines = file.readlines()

                is_save = False
                with open(temp_filepath, "w") as temp_file:
                    for line in lines:
                        split_string = line.split(" ")
                        # 判断是否与原始标签一致
                        if split_string[0] in keep_labels:
                            temp_file.write(line)
                            is_save = True

                if is_save:
                    os.remove(filepath)
                    os.rename(temp_filepath, filepath)
                else:
                    os.remove(filepath)
                    os.remove(temp_filepath)


def find_label():
    for directory in directorys:
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                # print(filepath)
                has_labels = False
                with open(filepath, "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        split_string = line.split(" ")
                        # 判断是否与原始标签一致
                        if filename == '2.txt':
                            print(split_string[0])
                        distinct_labels.add(split_string[0])

    print(distinct_labels)


# 去除不需要的标签
# remove_label()
# 替换标签号
# replace_label()
# 查看标签
find_label()
print("Done!")
