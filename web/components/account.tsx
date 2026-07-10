"use client";

import { useState } from "react";
import { signIn, signUp, useSession } from "@/lib/auth-client";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function Account() {
  const { data: session, isPending } = useSession();
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<"signup" | "signin">("signup");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  // Signed-in account controls live in the sidebar; header only handles signed-out.
  if (isPending || session?.user) return null;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr("");
    setBusy(true);
    const res =
      mode === "signup"
        ? await signUp.email({ email, password, name: name || email.split("@")[0] })
        : await signIn.email({ email, password });
    setBusy(false);
    if (res.error) setErr(res.error.message || "Something went wrong.");
    else setOpen(false);
  }

  return (
    <>
      <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
        Sign up to save chats
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle className="font-heading text-xl">
            {mode === "signup" ? "Save your chats" : "Welcome back"}
          </DialogTitle>
          <DialogDescription>
            {mode === "signup"
              ? "Create an account to keep your conversation history."
              : "Log in to pick up where you left off."}
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={submit} className="flex flex-col gap-3">
          {mode === "signup" && (
            <div className="grid gap-1.5">
              <Label htmlFor="name">Name</Label>
              <Input id="name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Your name" />
            </div>
          )}
          <div className="grid gap-1.5">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
            />
          </div>
          <div className="grid gap-1.5">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 8 characters"
            />
          </div>
          {err && <p className="text-sm text-destructive">{err}</p>}
          <Button type="submit" disabled={busy} className="mt-1">
            {busy ? "…" : mode === "signup" ? "Create account" : "Log in"}
          </Button>
        </form>
        <button
          type="button"
          className="text-xs text-muted-foreground transition-colors hover:text-foreground"
          onClick={() => {
            setMode(mode === "signup" ? "signin" : "signup");
            setErr("");
          }}
        >
          {mode === "signup" ? "Already have an account? Log in" : "Need an account? Sign up"}
        </button>
      </DialogContent>
      </Dialog>
    </>
  );
}
