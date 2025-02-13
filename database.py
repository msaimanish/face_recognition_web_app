import sqlite3

DB_PATH = "faces.db"

def create_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def insert_face(student_id, name, embedding):
    create_database()  # Ensure table exists before inserting
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        c.execute("INSERT INTO faces (student_id, name, embedding) VALUES (?, ?, ?)", 
                  (student_id, name, embedding))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"⚠️ Student ID {student_id} already exists. Skipping insert.")
        conn.close()
        return "Student ID already exists"
    
    conn.close()
    return "Success"

def get_all_faces():
    create_database()  # Ensure table exists before fetching
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT student_id, name, embedding FROM faces")
    faces = c.fetchall()
    conn.close()
    return faces

# Ensure the database is created on startup
create_database()
