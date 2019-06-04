#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define OFFSET 205;
#define STEER 1
#define THROTTLE 0
#define STEER_BYTE 35
#define THROTTLE_BYTE 38

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);             // Serial port to computer
  pwm.begin();
  pwm.setPWMFreq(50);
  int steer = 110 + OFFSET;
  int throttle = 102 + OFFSET;
  
  pwm.setPWM(STEER, 0, steer);
  pwm.setPWM(THROTTLE, 0, throttle);
  pinMode(5, OUTPUT);
  digitalWrite(5, HIGH);
}

void loop() {

  int steer = 110 + OFFSET;
  int throttle = 102 + OFFSET;
  int hash_function = 255;
  byte serial_read = 0;
  byte hash_write = 0;
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0 ){
    serial_read = Serial.read();
    if (serial_read == STEER_BYTE){
      
      while (Serial.available() < 1){ }
      steer = Serial.read();
      hash_function = hash_function + STEER_BYTE * steer;
      steer = steer + OFFSET;
      pwm.setPWM(STEER, 0, steer);
      while (Serial.available() < 1){ }
      if (serial_read == THROTTLE_BYTE){
        throttle = Serial.read();
        hash_function = hash_function - throttle;
        throttle = throttle + OFFSET;
        pwm.setPWM(THROTTLE, 0, throttle);
      }
    hash_function = hash_function % 255;
    hash_write = (byte) hash_function;
    Serial.write(hash_write);  
    }
  }
}
