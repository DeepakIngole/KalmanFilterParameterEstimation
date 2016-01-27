function z = ObservationMapping( w, x )
wt = zeros(13,14);
wt(:) = w(:);
y = zeros(size(x));
    for i = 1:length(x(1,:))
    y(:,i) = wt*[x(:,i);1];
    end
z = y(:);
end

