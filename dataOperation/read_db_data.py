# _*_coding:utf-8_*_
import cx_Oracle as oracle


def connection(user_name='bendan', password='bendan', service_name='orcl'):
    db = oracle.connect(user_name, password, service_name)
    return db


def select(connection, table_name, condition, *attributes):
    cur = connection.cursor()
    sql = 'select '
    i = 1
    for attribute in attributes:
        sql += attribute
        if i < len(attributes):
            sql += ', '
            i += 1
    sql += ' from '
    sql += table_name
    
    sql += condition
    
    print '打印查询语句：'
    print sql
    cur.execute(sql)

    

    
    file = open(table_name+'.txt', 'w')
    for result in cur:
        line = ''
        for ele in result:
            line += str(ele)
            line += ', '
        file.write(line+'\n')
    file.close()
    cur.close()


def select_sql(connection, sql):
    cur = connection.cursor()
    print sql
    cur.execute(sql)
    file = open('IMIS_3_DAY.txt', 'w')
    for result in cur:
        line = ''
        for ele in result:
            line += str(ele)
            line += ', '
        file.write(line+'\n')
    file.close()
    cur.close()


def select_sql2(connection, sql, file_name):
    cur = connection.cursor()
    print sql
    cur.execute(sql)
    file = open(file_name, 'w')
    for result in cur:
        line = ''
        for ele in result:
            line += str(ele)
            line += ', '
        file.write(line+'\n')
    file.close()
    cur.close()


def select_sql3(sql, file_name):
    db = connection()
    cur = db.cursor()
    print sql
    cur.execute(sql)
    file = open(file_name, 'w')
    for result in cur:
        line = ''
        for ele in result:
            line += str(ele)
            line += ', '
        file.write(line+'\n')
    file.close()
    cur.close()


def get_record_count(sql):
    db = connection()
    cur = db.cursor()
    print sql
    cur.execute(sql)
    count = 0
    for result in cur:
        count = int(result[0])
        break
    return count
    

def main():
    db = connection()
    # select(db, 'cruise_track', 'lon', 'lat', 'time')
    # select(db, 'AISDATA', " where id = '232002648'", 'id', 'time', 'lon', 'lat')
    sql = "select * from IMIS_7_10 t where t.obj_id = '792' order by t.time"
    file_name = 'geolife_data\MIS_7_10_792.txt'
    select_sql2(db, sql, file_name)
    db.close()
            
# main()
