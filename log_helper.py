import csv




class CsvLogger:
    def __init__(self, filename, head):
        self.filename = filename
        # 初始化时创建文件并写入标题
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(head)  # 举例的标题行

    def log(self, data):
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        
        
