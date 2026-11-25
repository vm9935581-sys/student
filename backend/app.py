
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load ML model
model_path = os.path.join(BASE_DIR, "student_model.pkl")

try:
    model = joblib.load(model_path)
    print("ML model loaded successfully.")
except Exception as e:
    model = None
    print("ML model loading error:", e)


# TiDB / Render database configuration
DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
    "database": os.environ.get("DB_NAME"),
    "port": int(os.environ.get("DB_PORT", 4000))
}


# Database connection
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            port=DB_CONFIG["port"],
            ssl_verify_cert=True,
            ssl_verify_identity=True
        )
        return connection

    except Error as e:
        print("MySQL connection error:", e)
        return None


# Common prediction function
def calculate_prediction(data):
    input_data = pd.DataFrame([{
        "attendance": float(data.get("attendance", 0) or 0),
        "previousMarks": float(data.get("previousMarks", 0) or 0),
        "assignmentMarks": float(data.get("assignmentMarks", 0) or 0),
        "internalMarks": float(data.get("internalMarks", 0) or 0),
        "studyHours": float(data.get("studyHours", 0) or 0),
        "completedAssignments": float(data.get("completedAssignments", 0) or 0),
        "previousFailures": float(data.get("previousFailures", 0) or 0)
    }])

    if model is None:
        raise Exception("ML model could not be loaded. Check student_model.pkl.")

    predicted_marks = float(model.predict(input_data)[0])
    predicted_marks = round(max(0, min(100, predicted_marks)), 2)

    if predicted_marks >= 75:
        performance = "Excellent"
        risk = "Low Risk"
    elif predicted_marks >= 60:
        performance = "Good"
        risk = "Medium Risk"
    elif predicted_marks >= 40:
        performance = "Average"
        risk = "Medium Risk"
    else:
        performance = "Needs Improvement"
        risk = "High Risk"

    return predicted_marks, performance, risk


# Common student details
def get_student_details(data):
    return (
        str(data.get("name", "")).strip(),
        str(data.get("rollNumber", "")).strip(),
        str(data.get("email", "")).strip(),
        str(data.get("phone", "")).strip(),
        int(data.get("age", 0) or 0),
        str(data.get("department", "")).strip(),
        int(data.get("semester", 0) or 0),
        float(data.get("attendance", 0) or 0),
        float(data.get("previousMarks", 0) or 0),
        float(data.get("assignmentMarks", 0) or 0),
        float(data.get("internalMarks", 0) or 0),
        float(data.get("studyHours", 0) or 0),
        float(data.get("completedAssignments", 0) or 0),
        float(data.get("previousFailures", 0) or 0)
    )


# Home
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Student Performance Prediction Backend is running!"
    })


# Database test
@app.route("/database-test", methods=["GET"])
def database_test():
    connection = get_db_connection()

    if connection is None:
        return jsonify({
            "success": False,
            "message": "Unable to connect to MySQL database."
        }), 500

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE()")
        database_name = cursor.fetchone()[0]

        cursor.close()
        connection.close()

        return jsonify({
            "success": True,
            "message": "MySQL database connected successfully!",
            "database": database_name
        })

    except Error as e:
        connection.close()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# Predict and insert student
@app.route("/predict", methods=["POST"])
def predict():
    connection = None
    cursor = None

    try:
        data = request.get_json() or {}
        details = get_student_details(data)

        name = details[0]
        roll_number = details[1]

        if not name or not roll_number:
            return jsonify({
                "success": False,
                "error": "Student name and roll number are required."
            }), 400

        connection = get_db_connection()

        if connection is None:
            return jsonify({
                "success": False,
                "error": "Unable to connect to MySQL database."
            }), 500

        cursor = connection.cursor()

        cursor.execute(
            "SELECT id FROM students WHERE roll_number = %s LIMIT 1",
            (roll_number,)
        )

        if cursor.fetchone():
            return jsonify({
                "success": False,
                "error": f"Student with roll number {roll_number} already exists."
            }), 409

        predicted_marks, performance, risk = calculate_prediction(data)

        query = """
            INSERT INTO students (
                name, roll_number, email, phone, age,
                department, semester, attendance,
                previous_marks, assignment_marks,
                internal_marks, study_hours,
                completed_assignments, previous_failures,
                predicted_marks, performance, risk
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s
            )
        """

        cursor.execute(query, details + (
            predicted_marks, performance, risk
        ))

        connection.commit()
        student_id = cursor.lastrowid

        return jsonify({
            "success": True,
            "studentId": student_id,
            "predictedMarks": predicted_marks,
            "performance": performance,
            "risk": risk,
            "message": "Prediction completed and student saved to MySQL."
        })

    except Exception as e:
        if connection:
            connection.rollback()

        print("Prediction error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# Get all students
@app.route("/students", methods=["GET"])
def get_students():
    connection = None
    cursor = None

    try:
        connection = get_db_connection()

        if connection is None:
            return jsonify({
                "success": False,
                "error": "Unable to connect to MySQL database."
            }), 500

        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                id, name, roll_number, email, phone, age,
                department, semester, attendance,
                previous_marks, assignment_marks,
                internal_marks, study_hours,
                completed_assignments, previous_failures,
                predicted_marks, performance, risk, created_at
            FROM students
            ORDER BY id DESC
        """)

        students = cursor.fetchall()

        formatted_students = []

        for student in students:
            formatted_students.append({
                "id": student["id"],
                "name": student["name"],
                "rollNumber": student["roll_number"],
                "email": student["email"],
                "phone": student["phone"],
                "age": student["age"],
                "department": student["department"],
                "semester": student["semester"],
                "attendance": student["attendance"],
                "previousMarks": student["previous_marks"],
                "assignmentMarks": student["assignment_marks"],
                "internalMarks": student["internal_marks"],
                "studyHours": student["study_hours"],
                "completedAssignments": student["completed_assignments"],
                "previousFailures": student["previous_failures"],
                "predictedMarks": student["predicted_marks"],
                "performance": student["performance"],
                "risk": student["risk"],
                "createdAt": str(student["created_at"])
            })

        return jsonify({
            "success": True,
            "students": formatted_students
        })

    except Exception as e:
        print("Get students error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# Update student
@app.route("/students/<int:student_id>", methods=["PUT", "OPTIONS"])
def update_student(student_id):
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    connection = None
    cursor = None

    try:
        data = request.get_json() or {}
        details = get_student_details(data)

        name = details[0]
        roll_number = details[1]

        if not name or not roll_number:
            return jsonify({
                "success": False,
                "error": "Student name and roll number are required."
            }), 400

        connection = get_db_connection()

        if connection is None:
            return jsonify({
                "success": False,
                "error": "Unable to connect to MySQL database."
            }), 500

        cursor = connection.cursor()

        cursor.execute("""
            SELECT id FROM students
            WHERE roll_number = %s AND id <> %s
            LIMIT 1
        """, (roll_number, student_id))

        if cursor.fetchone():
            return jsonify({
                "success": False,
                "error": f"Student with roll number {roll_number} already exists."
            }), 409

        predicted_marks, performance, risk = calculate_prediction(data)

        cursor.execute("""
            UPDATE students SET
                name = %s,
                roll_number = %s,
                email = %s,
                phone = %s,
                age = %s,
                department = %s,
                semester = %s,
                attendance = %s,
                previous_marks = %s,
                assignment_marks = %s,
                internal_marks = %s,
                study_hours = %s,
                completed_assignments = %s,
                previous_failures = %s,
                predicted_marks = %s,
                performance = %s,
                risk = %s
            WHERE id = %s
        """, details + (
            predicted_marks, performance, risk, student_id
        ))

        if cursor.rowcount == 0:
            return jsonify({
                "success": False,
                "error": "Student not found."
            }), 404

        connection.commit()

        return jsonify({
            "success": True,
            "studentId": student_id,
            "predictedMarks": predicted_marks,
            "performance": performance,
            "risk": risk,
            "message": "Student updated successfully."
        })

    except Exception as e:
        if connection:
            connection.rollback()

        print("Update student error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# Delete one student
@app.route("/students/<int:student_id>", methods=["DELETE", "OPTIONS"])
def delete_student(student_id):
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    connection = None
    cursor = None

    try:
        connection = get_db_connection()

        if connection is None:
            return jsonify({
                "success": False,
                "error": "Unable to connect to MySQL database."
            }), 500

        cursor = connection.cursor()

        cursor.execute(
            "DELETE FROM students WHERE id = %s",
            (student_id,)
        )

        if cursor.rowcount == 0:
            return jsonify({
                "success": False,
                "error": "Student not found."
            }), 404

        connection.commit()

        return jsonify({
            "success": True,
            "message": "Student deleted successfully."
        })

    except Exception as e:
        if connection:
            connection.rollback()

        print("Delete student error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# Delete all students
@app.route("/students", methods=["DELETE", "OPTIONS"])
def delete_all_students():
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    connection = None
    cursor = None

    try:
        connection = get_db_connection()

        if connection is None:
            return jsonify({
                "success": False,
                "error": "Unable to connect to MySQL database."
            }), 500

        cursor = connection.cursor()
        cursor.execute("DELETE FROM students")

        connection.commit()

        return jsonify({
            "success": True,
            "message": "All students deleted successfully."
        })

    except Exception as e:
        if connection:
            connection.rollback()

        print("Delete all students error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# Run Flask application
if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
