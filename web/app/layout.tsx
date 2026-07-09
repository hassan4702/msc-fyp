import type { Metadata } from "next";
import { Fraunces, Hanken_Grotesk } from "next/font/google";
import "./globals.css";

const sans = Hanken_Grotesk({ subsets: ["latin"], variable: "--font-sans" });
const heading = Fraunces({
  subsets: ["latin"],
  variable: "--font-heading",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Empath — Emotion-Aware Companion",
  description: "A chatbot that reads emotion from your face and words, and replies with that in mind.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${sans.variable} ${heading.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <div className="ambient" />
        {children}
      </body>
    </html>
  );
}
