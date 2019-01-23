class EarlyStopping(object):
    def __init__(self, patience=15, min_delta = 0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.patience_cnt = 0
        self.prev_loss_val = 200000
        self.patient_cum = 0
        
    
    def stop(self, loss_val):
        if(self.prev_loss_val - loss_val>self.min_delta):
            self.patience_cnt = 0
            self.prev_loss_val = loss_val
            
        else:
            self.patience_cnt += 1
            self.patient_cum +=1
            print('Patience count: ', self.patience_cnt)
            
        if(self.patience_cnt > self.patience or self.patient_cum > 400):
            return True
        else:
            return False
        
    