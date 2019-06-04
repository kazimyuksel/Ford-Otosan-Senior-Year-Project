clear
clc
theta_t = 0 %deg
theta_c = 85 %deg

v_front = 5 %m/s
time_step = 0.001 %s

L = 10 %m: trailer lenght

theta_t = theta_t * pi / 180;
theta_c = theta_c * pi / 180;
initial_position_x = 0;
initial_position_y = 0;

x_increment = v_front*cos(theta_c)*time_step;
position_increment = v_front*time_step;
x = initial_position_x:x_increment:(initial_position_x + 30);

%y eq
%bu kýsým yanlýþ, v_yatay sabit kabul ediliyor, sanki y de hiç deðiþmiyor
%gibi oldu ondan dolayý ilkin spacing fonksiyonu yazýlmalý.
y = tan(theta_c)*x;
plot(x,y)
hold on

b_x = initial_position_x - L*cos(theta_t);
b_y = initial_position_y + L*sin(theta_t);

b_new = b_x;
i = 1;
for i = 1:length(x)-1
    a = b_new - 100*position_increment;
    b = b_new + 100*position_increment;
    p = b_new;
    f = p;
    func_p = (f-x(i+1))^2 + (tan(pi-theta_t(i))*f+y(i)-tan(pi-theta_t(i))*x(i)-y(i+1))^2-L.^2;
    err = abs(func_p);
    while err > 1e-12
        f = a;
        func_a = (f-x(i+1))^2 + (tan(pi-theta_t(i))*f+y(i)-tan(pi-theta_t(i))*x(i)-y(i+1))^2-L.^2;
        f = b;
        func_b = (f-x(i+1))^2 + (tan(pi-theta_t(i))*f+y(i)-tan(pi-theta_t(i))*x(i)-y(i+1))^2-L.^2;
        
        if func_a*func_p<0
            b = p;
        else
            a = p;
        end
        p = (a + b)/2;
        f = p;
        func_p = (f-x(i+1))^2 + (tan(pi-theta_t(i))*f+y(i)-tan(pi-theta_t(i))*x(i)-y(i+1))^2-L.^2;
        err = abs(func_p);
        %disp(err)
    end
    b_x = [b_x f];
    b_y_cal = tan(pi-theta_t(i))*f+y(i)-tan(pi-theta_t(i))*x(i);
    b_y = [b_y b_y_cal];
    angle = atan((b_y_cal-y(i+1))/(x(i+1)-f));
    theta_t = [theta_t angle];
    b_new = f;
    
end
plot(b_x,b_y)
axis equal

