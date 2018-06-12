# _*_coding:utf-8_*_

import MySQLdb


def db_connection(host_name='localhost', user_name='root', password='root', database='diditech', charset='utf8'):
    conn = MySQLdb.connect(host=host_name, user=user_name, passwd=password, db=database, charset=charset)
    return conn


def select_sql(sql):
    db = db_connection()
    cursor = db.cursor()
    try:
        cursor.execute(sql)
        db.commit()
        # result = cursor.fetchall()
        # for row in result:
        #     print row[1]
    except Exception:
        print 'error'
    db.close()


def get_record_count(sql):
    db = db_connection(host_name='localhost', user_name='root', password='root', database='stdatamining', charset='utf8')
    cur = db.cursor()
    print sql
    cur.execute(sql)
    results = cur.fetchall()
    count = 0
    for result in results:
        count = int(result[0])
        break
    return count


# select_sql("DELETE FROM 2016_01_01_order_data_copy WHERE order_id='0262fdae65c4b8c46876ad553683d940'")

# get_record_count("SELECT COUNT(*) FROM GEOLIFE_153 t WHERE  t.time_date < '2008-10-01 23:59:59' "
#                  "AND t.time_date > '2008-10-01 00:00:00' ORDER BY time_date")
