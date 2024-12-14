require 'socket'
starttime = Process.clock_gettime(Process::CLOCK_MONOTONIC)
s = TCPSocket.new 'localhost' , 8989

s.write("../testfiles/#{ARGV[0]}.cpp\n")

s.each_line do |line|
    # puts line
end

s.close

endtime = Process.clock_gettime(Process::CLOCK_MONOTONIC)
elapsed = endtime - starttime
puts "ELAPSED:  #{elapsed} (#{ARGV[0]})" 