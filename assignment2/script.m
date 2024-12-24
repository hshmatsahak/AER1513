% Load the dataset
load('dataset2.mat');

% Initialize variables
T = 0.1; % Sampling period
num_timesteps = length(t);
x_est = zeros(num_timesteps, 3); % [x, y, theta] estimates
P_est = zeros(3, 3, num_timesteps); % Covariance matrix

% Initial conditions
x_est(1, :) = [1, 1, 0.1]; % Poor initial condition for (b)
P_est(:, :, 1) = diag([1, 1, 0.1]); % Initial covariance matrix

% Define measurement noise covariances
R_max_values = [5, 3, 1]; % rmax values to test

% EKF loop
for i = 1:length(R_max_values)
    r_max = R_max_values(i);
    
    for k = 2:num_timesteps
        % Prediction step
        x_prev = x_est(k-1, :)';
        v_k = v(k);
        omega_k = om(k);
        
        % State transition
        F_k = [1, 0, -v_k * sin(x_prev(3)) * T;
               0, 1, v_k * cos(x_prev(3)) * T;
               0, 0, 1];
        
        x_pred = x_prev + T * [v_k * cos(x_prev(3));
                               v_k * sin(x_prev(3));
                               omega_k];
                           
        % Covariance prediction
        Q = diag([0.1, 0.1, 0.01]); % Process noise
        P_pred = F_k * P_est(:, :, k-1) * F_k' + Q;
        
        % Update step - consider only measurements within range
        visible_landmarks = find(r(k, :) < r_max & r(k, :) > 0);
        
        for landmark = visible_landmarks
            % Measurement model
            lx = l(landmark, 1);
            ly = l(landmark, 2);
            
            % Calculate expected measurement
            dx = lx - x_pred(1);
            dy = ly - x_pred(2);
            r_pred = sqrt(dx^2 + dy^2);
            phi_pred = atan2(dy, dx) - x_pred(3);
            
            % Measurement Jacobian
            H_k = [-dx/r_pred, -dy/r_pred, 0;
                   dy/(dx^2 + dy^2), -dx/(dx^2 + dy^2), -1];
               
            % Measurement noise
            R = diag([r_var(k), b_var(k)]);
            
            % Kalman gain
            S_k = H_k * P_pred * H_k' + R;
            K_k = P_pred * H_k' / S_k;
            
            % Update state estimate and covariance
            z_k = [r(k, landmark); b(k, landmark)];
            z_pred = [r_pred; phi_pred];
            x_pred = x_pred + K_k * (z_k - z_pred);
            P_pred = (eye(3) - K_k * H_k) * P_pred;
        end
        
        % Store results
        x_est(k, :) = x_pred';
        P_est(:, :, k) = P_pred;
    end
    
    % Plot results for each variable and each r_max
    figure;
    for j = 1:3
        subplot(3, 1, j);
        if j == 1
            plot(t, x_est(:, 1) - x_true, 'b', 'LineWidth', 1.5); % x error
            hold on;
            plot(t, 3 * sqrt(squeeze(P_est(1, 1, :))), 'r--');
            plot(t, -3 * sqrt(squeeze(P_est(1, 1, :))), 'r--');
            ylabel('Error in x');
        elseif j == 2
            plot(t, x_est(:, 2) - y_true, 'b', 'LineWidth', 1.5); % y error
            hold on;
            plot(t, 3 * sqrt(squeeze(P_est(2, 2, :))), 'r--');
            plot(t, -3 * sqrt(squeeze(P_est(2, 2, :))), 'r--');
            ylabel('Error in y');
        else
            plot(t, x_est(:, 3) - th_true, 'b', 'LineWidth', 1.5); % theta error
            hold on;
            plot(t, 3 * sqrt(squeeze(P_est(3, 3, :))), 'r--');
            plot(t, -3 * sqrt(squeeze(P_est(3, 3, :))), 'r--');
            ylabel('Error in \theta');
        end
        xlabel('Time [s]');
        legend('Error', '3\sigma Envelope');
    end
end
