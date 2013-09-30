library(SASxport)

test.str1 <- intToUtf8(1:100) # R intToUtf8 omits ascii 0
test.str2 <- intToUtf8(101:127)

df <- data.frame(X=c(test.str1, test.str2))
print(df)
print(summary(df))
print(df$X)
print(df$X[1])
print(df$X[2])
write.xport(df, file="strings.xpt", autogen.formats=F)

df.vals <- data.frame(X=c(-1000:1000, pi ^ (-30:30), -pi ^ (-30:30)))

write.xport(df.vals, file="known_values.xpt")
