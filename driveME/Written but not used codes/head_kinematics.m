v = 10; %m/s
a = -15;
L = 3.6;
theta_heading = pi/4;
theta_steer = pi/4;
x=0;
y=0;
x_new = x;
y_new = y;
del_t = 0.0001;
c_x = x+0.7*cos(theta_heading);
c_y = y+0.7*sin(theta_heading);
vx=v*cos(theta_heading);
vy=v*sin(theta_heading);
for t = 0:del_t:1
   vx = vx+del_t*(a*cos(theta_heading)-v*v*tan(theta_steer)*sin(theta_heading)/L);
   vy = vy+del_t*(a*sin(theta_heading)+v*v*tan(theta_steer)*cos(theta_heading)/L);
   theta_heading = theta_heading + del_t*v*tan(theta_steer)/L;

   x_new = x_new + del_t*vx;
   y_new = y_new + del_t*vy;
   
   c_x = [c_x (x_new+0.7*cos(theta_heading))];
   c_y = [c_y (y_new+0.7*sin(theta_heading))];
   
   x = [x x_new];
   y = [y y_new];
   v = v+ a*del_t
   if v > 0 && v<0.01
       theta_steer = - theta_steer
   end
end
scatter(x,y)
axis equal
hold on
scatter(c_x,c_y)