% function [] = double_pend_fancy_sim(T, Y, R, L1, L2, skip, words);
%
%
% Dan Davison
% Last modified Apr 17 2019
% Does fancy graphical simulation of the 2-link inverted pendulum.
%
% Inputs:  T = time vector (column vector)
%          Y = plant data        (col 1 = w, col 2 = theta1, col 3 = theta2)
%          R = reference signals (col 1 = w, col 2 = theta1, col 3 = theta2)
%          L1 and L2 = lengths of the two links
%          skip = integer used to make plot less crowded (e.g., =2 implies plot every 2nd data point only)
%                 Note that "skip" only has an effect if the 'f' option, discussed below, is used.
%          words = title of the plot (optional argument)
%
% Execution:  The user is prompted with 3 options:
%          d = dynamic simulation, i.e., actually show how the system changes over time
%          f = skip the dynamic simulation, and draw a single picture showing how the 
%              system moves about over time [the "skip" argument to the program can be
%              adjusted to make the drawing less crowded]
%          s = skip everything [handy if you are calling this function from another
%              function and don't want to spend time watching the simulation]
%



function [] = double_pend_fancy_sim(T, Y, R, L1, L2, skip, words);

savemovie=0;  % manually set to 1 to save dynamic sim as a mp4 file

if nargin < 6
    error('Need to have at least 6 input arguments')
end

deltaT = T(2)-T(1);

if savemovie  % ddd Apr 3 2019
   disp('*** saving as a mp4 file ***')
   vidObj = VideoWriter('P6(b)_two_rod_movie.mp4','MPEG-4'); set(vidObj,'Quality',100);
   open(vidObj);
end

% figure out how to scale so the whole simulation fits on one screen and looks good

xmin = min(Y(:,1))+1.1*(-L1-L2);
xmax = max(Y(:,1))+1.1*(L1+L2);
ymin = 1.1*(-L1-L2);
ymax = 1.1*(L1+L2);

ymax = max(ymax, (xmax-xmin)+ymin); 
minL = min(L1,L2);
width = minL/4;  % width of the 'cart' is twice this value


% c = input('Hit: d for dynamic simulation, f for final plot, s for skip simulation...','s');
% c=c(1);
c = 'f';
if c=='s'
    return
end

if c=='d'
    dynamic = 1;
    skipterm = 1;  % don't skip anything in the dynamic plot
    if savemovie
        skipterm = 2;  % makes the movie file run faster DDD
    end
else  
    dynamic = 0;
    skipterm = skip;  % skip some of the data as the user specifies
end

for i = 1:skipterm:length(T)
    
   xposition = Y(i,1);
   theta1 = Y(i,2);
   theta2 = Y(i,3);
   
   xposition_ref = R(i,1);
   theta1_ref = R(i,2);
   theta2_ref = R(i,3);
   
   % draw cart as a thick line
   h=plot([xposition-width xposition+width],[0 0],'k'); set(h,'linewidth',5);
   hold on
   
   % draw link 1
   t1 = xposition+L1*sin(theta1);
   t2 = L1*cos(theta1);
   h1=plot([xposition t1],[0 t2],'b'); set(h1, 'linewidth',3);
   
   % draw link 2
   h2=plot([t1 t1+L2*sin(theta2)],[t2 t2+L2*cos(theta2)],'g'); set(h2, 'linewidth',3);
   
   if dynamic
 
      % also draw the reference shape in red
      t1_ref = xposition_ref+L1*sin(theta1_ref);
      t2_ref = L1*cos(theta1_ref);
      plot([xposition_ref-width xposition_ref+width],[0 0],'r');
      plot([xposition_ref t1_ref],[0 t2_ref],'r');
      plot([t1_ref t1_ref+L2*sin(theta2_ref)],[t2_ref t2_ref+L2*cos(theta2_ref)],'r');  
      
      % label plot
      title(sprintf('Time = %f seconds', T(i)'))
      hold off    
      axis([xmin xmax ymin ymax]);
      axis('square')
      xlabel('Base position (m)')
      pause(deltaT)
   else  
      % keep holding plot
      h.Color(4) = ((i)/length(T) * 1.0); % compute alpha
      h1.Color(4) = ((i)/length(T) * 1.0); % compute alpha
      h2.Color(4) = ((i)/length(T) * 1.0); % compute alpha
   end % if
   
    if savemovie % ddd Apr 3 2019
      currFrame = getframe(gcf);
      writeVideo(vidObj,currFrame);
    end
   
 end % for

axis([xmin xmax ymin ymax]);
axis('square')
if nargin==7
  title(words);
end
hold off

if savemovie % save the movie DDD
   close(vidObj)
end
 

