from deepface import DeepFace
import json

def face_verify(img1,img2):
    try:
        result_dict = DeepFace.verify(img1_path = img1,img2_path = img2)

        with open('result.json', 'w') as file:
            json.dump(result_dict, file,indent=4,ensure_ascii=False)            
        
        # return result_dict

        if result_dict.get('verified'):
            return 'Active'
        
        return 'Failed'
        
    except Exception as e:
        return e


def face_recognition():
    try:
        result = DeepFace.find(img_path='faces/em1.jpg',db_path='emma')
        
        return result
    except Exception as e:
        return e
    


def face_analize():
    try:
        result_dict = DeepFace.analyze(img_path='test.jpg',actions=['age','gender','race','emotion'])

        with open('face_analize.json', 'w') as file:
            json.dump(result_dict, file,indent=4,ensure_ascii=False)
        
        return result_dict

    except Exception as e:
        return e


# print(face_verify('faces/hr1.jpg','faces/em2.jpg'))
# print(face_recognition())
print(face_analize())