clear
clc
pathname = 'G:\signal\time_cut';
pathnew = 'G:\signal\new_time_cut\';
img_path_list = dir(pathname);
img_num = length(img_path_list);
for j = 3:img_num
    all_img = dir(strcat(pathname,'\',img_path_list(j).name, '\*.mat'));
    filelength = length(all_img);
    labelname = img_path_list(j).name;   
    mkdir(pathnew,labelname)
    for k = 1:filelength
        ch1 = load(strcat(pathname,'\',img_path_list(j).name,'\',all_img(k).name));
        ECG_data = ch1.data;
        datalength = length(ECG_data);
        label = 0;
        if datalength >= 38400 && datalength < 40000
            label = 128;
        elseif datalength >= 75000 && datalength < 85000
            label = 250;
        elseif datalength >= 150000 && datalength < 160000  
            label = 500;
        end
        
        [f,g]=xcorr(ECG_data,'coeff');
        L = floor(length(f) / 2) + 2;
        xcorrdata = f(1:L);
        [pks_max,locs_max] = findpeaks(xcorrdata);
        if size(pks_max,2) > 2
            pks_max_sort = sort(pks_max,'descend');
            P1 = pks_max_sort(1);
            P2 = pks_max_sort(2);
            P1_position = find(xcorrdata==P1);
            P2_position = find(xcorrdata==P2);
            diff = abs(P1_position - P2_position(1));
            if label == 128
                if diff > 50 && diff < 140
                    save(strcat(pathnew,labelname,'\', all_img(k).name), 'ECG_data');
                end
           elseif label == 250
                if diff > 90 && diff < 250
                    save(strcat(pathnew,labelname,'\', all_img(k).name), 'ECG_data');
                end
           elseif label == 500
                if diff > 150 && diff < 450
                    save(strcat(pathnew,labelname,'\', all_img(k).name), 'ECG_data');
                end
            end
%             sprintf('%s', labelname)
%             sprintf('%.2f',diff)
        end
    end
end



































