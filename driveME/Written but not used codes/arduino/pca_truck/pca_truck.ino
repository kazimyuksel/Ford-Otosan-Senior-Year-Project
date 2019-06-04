#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// called this way, it uses the default address 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
// you can also call it with a different address you want
//Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x41);

// Depending on your servo make, the pulse width min and max may vary, you 
// want these to be as small/large as possible without hitting the hard stop
// for max range. You'll have to tweak them as necessary to match the servos you
// have!
#define SERVOMIN  1200 // this is the 'minimum' pulse length count (out of 20000)
#define SERVOMAX  1800 // this is the 'maximum' pulse length count (out of 20000)

// our servo # counter
uint8_t servonum = 0;
uint32_t pulselength;
uint16_t on_lenght = 300;
void setup() {
  uint16_t on_lenght = 300;
  Serial.begin(9600);
  Serial.println("16 channel Servo test!");

  pwm.begin();
  pulselength = 1000000;   // 1,000,000 us per second
  pulselength /= 50;   // 60 Hz
  Serial.print(pulselength); Serial.println(" us per period"); 
  pulselength /= 4096;  // 12 bits of resolution
  Serial.print(pulselength); Serial.println(" us per bit"); 
  pwm.setPWMFreq(50);  // Analog servos run at ~60 Hz updates
  delay(1000);
  pwm.setPWM(servonum, 0, on_lenght);
}

void loop() {
  Serial.print(on_lenght); Serial.println(" /4096");
  pwm.setPWM(servonum, 0, on_lenght);
  delay(1500);
  on_lenght = on_lenght - 1;
}
