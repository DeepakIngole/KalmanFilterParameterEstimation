function wout = Dynamics( t, w , sigma)
wt = zeros(13,14);
wt(:) = w(:);
wo = wt + normrnd( 0, sigma , size(wt));
wout = w(:);
end

