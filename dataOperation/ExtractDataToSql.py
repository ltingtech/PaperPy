# _*_coding:utf-8_*_


def data_to_sql(data_file, sql_file):
    output_file = open(data_file, 'r')
    input_file = open(sql_file, 'w')
    input_file.write("create table roman_taxi (\nTRAID varchar(10) ,\nLONGITUDE number(9,6) ,\n" +
	"LATITUDE number(9,6) ,\ntime_date varchar(20)\n);\n")
    for line in output_file:
        content = line.split(';')
        obj_id = content[0]
        time_date = content[1].split('.')[0]
        # t1 = time.strptime(time_date, "%Y-%m-%d %H:%M:%S")
        # time_mils = int(time.mktime(t1) * 1000)
        point = content[2].split('\n')[0]
        point_lon = point.split(' ')[0].split('(')[1]
        point_lat = point.split(' ')[1].split(')')[0]
        sql_base = 'insert into roman_taxi (TRAID, LONGITUDE, LATITUDE, TIME_DATE) '
        sql_value = "values (" + "'" + obj_id + "','" + point_lon + "','" + point_lat + "','" + time_date + "');\n"
        sql = sql_base + sql_value
        input_file.write(sql)
    input_file.write('commit;')
    output_file.close()
    input_file.close()


data_to_sql("E:\\Science\\data\\taxi_february\\taxi_february.txt", "E:\\Science\\data\\taxi_february\\sql.txt")