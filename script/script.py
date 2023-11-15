def process_file(input_filename, output_filename):
    try:
        # 打开输入文件以读取内容
        with open(input_filename, 'r') as input_file:
            lines = input_file.readlines()

        # 处理每一行：替换换行符为空格，处理连字符结尾
        processed_lines = []
        for line in lines:
            line = line.strip()  # 删除前导和尾随空格
            if not line.endswith('-'):
                line = line + ' '
            else:
                line = line[:-1]  # 删除连字符 '-'
            processed_lines.append(line)

        # 将处理后的内容写入输出文件
        with open(output_filename, 'w') as output_file:
            output_file.write(''.join(processed_lines))

        print(f"处理完成，结果已写入 {output_filename}")
    except FileNotFoundError:
        print(f"文件 {input_filename} 不存在。")
    except Exception as e:
        print(f"发生错误：{str(e)}")

# 调用函数并指定输入和输出文件名
input_filename = "input.txt"
output_filename = "output.txt"
process_file(input_filename, output_filename)
