function r = psudogen(n,p)

x1 = [];
for i=1:15000
    if i == 1
        x1(i) = 1;
    elseif i <= 31
        x1(i) = 0;
    else
        x1(i) = mod(x1(i-31) + x1(i-28),2);
    end
end


%x2 gen
x2 = [];

for i=1:15000
    if i <= 31
        if p == 0
            if i == 18
                x2(i) = 1;
            else 
                x2(i) = 0;
            end
        elseif p == 1
            if i == 19
                x2(i) = 1;
            else
                x2(i) = 0;
            end
        elseif p == 2
            if i == 20
                x2(i) = 1;
            else
                x2(i) = 0;
            end
        elseif p == 3
            if i == 21
                x2(i) = 1;
            else
                x2(i) = 0;
            end
        end
    else
        x2(i) = mod(x2(i-31)+x2(i-30)+x2(i-29)+x2(i-28),2);
    end
end

%c(n) gen
Nc = 1600;
c = [];
for i=1:10000
    c(i) = mod(x1(i+Nc)+x2(i+Nc),2);
end

%r(n) gen
for i = 1:n
    r(i) = 1/sqrt(2)*(1-2*c(2*i))+1i*1/sqrt(2)*(1-2*c(2*i+1));
end
