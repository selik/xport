## Generates two sample SAS XPORT files for testing `xport` package.

library(SASxport)

# -- Produce a single-column test file with some strings with
#    characters throughout ascii.

test.str1 <- intToUtf8(1:100) # R has some strange behavior wrt string 0x0
test.str2 <- intToUtf8(101:127)

df <- data.frame(X=c(test.str1, test.str2))
write.xport(df, file="strings.xpt", autogen.formats=F)

# -- Produce a single-column test file with some floating point
#    numbers of various magnitudes and plenty of irrationality.

df.vals <- data.frame(X=c(-1000:1000, pi ^ (-30:30), -pi ^ (-30:30)))

write.xport(df.vals, file="known_values.xpt")

# -- Produce a multi-column test file with some strings with
#    characters throughout ascii.

strings <- strsplit("This is one time where television really fails to capture the true excitement of a large squirrel predicting the weather.", " ")[[1]]
floats <- 1:length(strings)

df.multi <- data.frame(list(X=strings, Y=floats))

write.xport(df.multi, file="multi.xpt", autogen.formats=F)
