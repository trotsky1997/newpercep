    import cv2
    from facelib import FaceRecognizer, FaceDetector
    from facelib import update_facebank, load_facebank, special_draw, get_config
 
    conf = get_config()
    detector = FaceDetector()
    face_rec = FaceRecognizer(conf)
    face_rec.model.eval()
    
    # set True when you add someone new 
    update_facebank_for_add_new_person = False
    if update_facebank_for_add_new_person:
        targets, names = update_facebank(conf, face_rec.model, detector)
    else:
        targets, names = load_facebank(conf)

    image = cv2.imread(your_path)
    faces, boxes, scores, landmarks = detector.detect_align(image)
    results, score = face_rec.infer(conf, faces, targets)
    print(names[results.cpu()])
    for idx, bbox in enumerate(boxes):
        special_draw(image, bbox, landmarks[idx], names[results[idx]+1], score[idx])