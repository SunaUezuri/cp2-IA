
import cv2
import mediapipe as mp
import numpy as np
import time


class HandDetector:
    """
    Classe para detecção e rastreamento de mãos usando MediaPipe
    
    Args:
        mode: Se True, detecta em cada frame. Se False, usa tracking
        max_hands: Número máximo de mãos a detectar
        detection_confidence: Confiança mínima para detecção
        tracking_confidence: Confiança mínima para rastreamento
    """
    
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
    def find_hands(self, img, draw=True):
        """
        Detecta mãos na imagem
        
        Args:
            img: Imagem BGR do OpenCV
            draw: Se True, desenha os landmarks na imagem
            
        Returns:
            img: Imagem com landmarks desenhados (se draw=True)
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    # Desenha conexões
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw_styles.get_default_hand_landmarks_style(),
                        self.mp_draw_styles.get_default_hand_connections_style()
                    )
        
        return img
    
    def find_position(self, img, hand_no=0, draw=True):
        """
        Encontra a posição dos landmarks de uma mão específica
        
        Args:
            img: Imagem BGR do OpenCV
            hand_no: Índice da mão (0 para primeira, 1 para segunda, etc)
            draw: Se True, desenha círculos nos landmarks
            
        Returns:
            landmark_list: Lista de [id, x, y] para cada landmark
        """
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                my_hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([id, cx, cy])
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        return landmark_list
    
    def fingers_up(self, landmark_list, hand_no=0):
        """
        Detecta quais dedos estão levantados (versão melhorada)
        
        Args:
            landmark_list: Lista de landmarks da mão
            hand_no: Índice da mão para determinar se é direita ou esquerda
            
        Returns:
            fingers: Lista de 5 elementos (0 ou 1) para cada dedo
                    [polegar, indicador, médio, anelar, mindinho]
        """
        fingers = []
        
        if len(landmark_list) != 0:
            tip_ids = [4, 8, 12, 16, 20]
            
            hand_type = self.get_hand_type(hand_no)
            
            thumb_distance = abs(landmark_list[4][1] - landmark_list[2][1])
            wrist_width = abs(landmark_list[5][1] - landmark_list[17][1])
            thumb_threshold = wrist_width * 0.3  
            
            if hand_type == "Right":
                if thumb_distance > thumb_threshold and landmark_list[tip_ids[0]][1] < landmark_list[tip_ids[0] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if thumb_distance > thumb_threshold and landmark_list[tip_ids[0]][1] > landmark_list[tip_ids[0] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            
            wrist_to_middle = abs(landmark_list[0][2] - landmark_list[9][2])
            threshold = wrist_to_middle * 0.1  
            
            for id in range(1, 5):
                tip_y = landmark_list[tip_ids[id]][2]
                middle_y = landmark_list[tip_ids[id] - 2][2]
                
                if tip_y < middle_y - max(threshold, 10):
                    fingers.append(1)
                else:
                    fingers.append(0)
        
        return fingers
    
    def get_hand_type(self, hand_no=0):
        """
        Determina se a mão é direita ou esquerda
        
        Args:
            hand_no: Índice da mão
            
        Returns:
            hand_type: "Right" ou "Left"
        """
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_handedness):
                return self.results.multi_handedness[hand_no].classification[0].label
        return None


def main():
    """Função principal para executar o detector de mãos"""
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  
    cap.set(4, 720)   
    
    detector = HandDetector(max_hands=2, detection_confidence=0.7)
    
    prev_time = 0
    
    print("=" * 60)
    print("RECONHECIMENTO DE MÃOS COM MEDIAPIPE")
    print("=" * 60)
    print("Pressione 'q' para sair")
    print("Pressione 'h' para alternar entre mostrar/ocultar landmarks")
    print("=" * 60)
    
    show_landmarks = True
    
    while True:
        success, img = cap.read()
        
        if not success:
            print("Erro ao capturar frame da webcam")
            break
        
        img = cv2.flip(img, 1)
        
        img = detector.find_hands(img, draw=show_landmarks)
        
        if detector.results.multi_hand_landmarks:
            num_hands = len(detector.results.multi_hand_landmarks)
            
            for hand_no in range(num_hands):
                landmark_list = detector.find_position(img, hand_no, draw=False)
                
                if len(landmark_list) != 0:
                    fingers = detector.fingers_up(landmark_list, hand_no)
                    num_fingers = fingers.count(1)
                    hand_type = detector.get_hand_type(hand_no)
                    
                    wrist_x, wrist_y = landmark_list[0][1], landmark_list[0][2]
                    
                    text_y = 50 + (hand_no * 100)
                    cv2.putText(img, f"Mao {hand_no + 1}: {hand_type}", 
                               (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (0, 255, 0), 2)
                    cv2.putText(img, f"Dedos levantados: {num_fingers}", 
                               (10, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    cv2.putText(img, f"Dedos: {fingers}", 
                               (10, text_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Nenhuma mao detectada", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, f"FPS: {int(fps)}", 
                   (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 0), 2)
        
        cv2.putText(img, "Pressione 'q' para sair | 'h' para landmarks", 
                   (img.shape[1] - 550, img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Reconhecimento de Maos - MediaPipe", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nEncerrando...")
            break
        elif key == ord('h'):
            show_landmarks = not show_landmarks
            status = "ativados" if show_landmarks else "desativados"
            print(f"Landmarks {status}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Programa finalizado!")


if __name__ == "__main__":
    main()